import pandas as pd
import numpy as np
import sklearn.utils.class_weight as cw
import scipy.sparse as sp
import tensorflow as tf
from sklearn.model_selection import train_test_split


class DataLoader:

    def __init__(self, config=None):
        self.config = config


    def loadGarph(self):

        gene_graph = pd.read_csv(self.config.graph_path, sep="\t")
        gene_mapping = pd.read_csv(self.config.graph_mapping, sep="\t")

        # mapping genes
        gene_graph =gene_graph.iloc[:, 0:2]
        gene_graph.columns = ["target", "source"]
        gene_mapping = dict(zip(gene_mapping['preferredName'], gene_mapping['queryItem']))
        gene_graph["target"] = gene_graph["target"].apply(lambda name: gene_mapping[name])
        gene_graph["source"] = gene_graph["source"].apply(lambda name: gene_mapping[name])

        self.gene_graph = gene_graph


    def loadData(self, type=None):

        self.datatype = type

        if type == 'cell_line':
            self.exp = pd.read_csv('_data/cell_line/' + self.config.cell_line_omic + "/" + "exp.csv")
            self.phe = pd.read_csv('_data/cell_line/' + self.config.cell_line_omic + "/" + "phe.csv")
            drug = pd.read_csv("_data/cell_line/cell_drug.txt", sep='\t')
            self.drug = drug.iloc[:, 0].to_list()


        if type == 'scSeq':
            self.exp = pd.read_csv('_data/scRNA/' + self.config.sc_cancer + "/" + "exp.csv")
            self.phe = pd.read_csv('_data/scRNA/' + self.config.sc_cancer + "/" + "phe.csv")

        if type == 'bulk':
            self.exp = pd.read_csv('_data/bulk/' + "exp.csv")
            self.phe = pd.read_csv('_data/bulk/' + "phe.csv")


    def preprocess(self, cancer=None):

        if self.datatype == 'cell_line':
            self.preprocess_cell_line()

        if self.datatype == 'scSeq':
            self.preprocess_scSeq()

        if self.datatype == 'bulk':
            self.preprocess_bulk(cancer=cancer)


    def string_to_numeric(self, dat):
        item = dat.unique().tolist()
        item_index = {name: id for id, name in enumerate(item)}
        return item_index

    def preprocess_graph(self, X):
        # preprocess graph
        genes = [i.split("_")[0] for i in X.columns if not "phe" in i]
        genes = np.unique(genes)
        genes = [item.replace('.', '-') for item in genes]
        item_index = {name: id for id, name in enumerate(genes)}

        gene_graph = self.gene_graph
        self.gene_dict = item_index
        gene_graph["source"] = gene_graph["source"].apply(lambda name: item_index[name])
        gene_graph["target"] = gene_graph["target"].apply(lambda name: item_index[name])

        gene_graph['value'] = 1
        adj = sp.coo_matrix((gene_graph.value, (gene_graph.target, gene_graph.source)),
                            shape=(len(item_index), len(item_index))).toarray()
        adj = adj + np.transpose(adj)
        adj = sp.coo_matrix(adj + sp.eye(adj.shape[0])).toarray()
        self.adj = adj

    def calculate_class_weight(self, X):
        y = X["phe_types"]
        class_weight = cw.compute_class_weight(class_weight='balanced',
                                               classes=np.unique(y),
                                               y=y)
        class_weight = {item: id for item, id in enumerate(class_weight)}
        self.class_weight = class_weight

    def preprocess_bulk(self, cancer=None):
        X = self.exp.drop(self.exp.columns[[0]], axis=1)
        phe = self.phe
        X['phe_drugType'] = phe['DRUG_TYPE']
        X['phe_cancer'] = phe['CANCER_TYPE'].to_numpy()
        X["phe_os"] = phe["OS_STATUS"].to_numpy()
        X["phe_os_time"] = phe["OS_MONTHS"].to_numpy()
        X["phe_id"] = phe["SAMPLE_ID"]
        X =X.loc[X.phe_drugType != 'Combo',]
        if not cancer == 'All':
            X = X.loc[X.phe_cancer == cancer,]
        self.id = X["phe_id"].tolist()
        self.X = X
        self.preprocess_graph(X)

    def preprocess_scSeq(self):
        X = self.exp.drop(self.exp.columns[[0]], axis=1)
        phe = self.phe

        if self.config.sc_cancer == 'lung':
            X["phe_types"] = phe['V4'].map(self.string_to_numeric(phe['V4']))
            self.id = phe['V1'].tolist()

        if self.config.sc_cancer == 'oscc':
            X["phe_types"] = phe['cluster']
            X["phe_id"] = phe['groups']
            X = X.loc[X.phe_types != 'Holiday',]
            X["phe_types"] = X["phe_types"].map(self.string_to_numeric(X["phe_types"]))
            self.id = X['phe_id'].tolist()

        self.preprocess_graph(X)
        self.calculate_class_weight(X)
        self.X = X

    def preprocess_cell_line(self):

        X = self.exp.drop(self.exp.columns[[0]], axis=1)

        X["phe_types"] = self.phe[self.config.cell_line_drug].to_numpy()
        X = X.loc[X.phe_types != 100,]

        # preprocess graph
        self.preprocess_graph(X)

        # calculate class weight
        self.calculate_class_weight(X)

        # function for fusion testing

        if self.config.cell_line_fusion == 'mut':
            X.loc[:, [item for item in X.columns if item.split('_')[1] == 'mut']] = 0

        if self.config.cell_line_fusion == 'cnv':
            X.loc[:, [item for item in X.columns if item.split('_')[1] == 'gain']] = 0
            X.loc[:, [item for item in X.columns if item.split('_')[1] == 'loss']] = 0

        if self.config.cell_line_fusion == 'exp':
            X.loc[:, [item for item in X.columns if item.split('_')[1] == 'exp']] = 0

        if self.config.cell_line_fusion == 'sv':
            X.loc[:, [item for item in X.columns if item.split('_')[1] == 'sv']] = 0

        self.X = X

    def process_for_model_input(self, dat):
        y = dat["phe_types"]
        x = dat.loc[:, [item for item in dat.columns if not 'phe' in item]]
        x = tf.constant(x, dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)

        return x, y

    def splitData(self, random_state=None):
        train, test = train_test_split(self.X, test_size=0.2, random_state=random_state)
        train, val = train_test_split(train, test_size=0.2, random_state=123)

        if self.datatype=='bulk':
            self.x_train, self.ytime_train, self.yevent_train = self.sortData(train)
            self.x_test, self.ytime_test, self.yevent_test = self.sortData(test)
            self.x_val, self.ytime_val, self.yevent_val = self.sortData(val)
        else:
            self.x_train, self.y_train = self.process_for_model_input(train)
            self.x_test, self.y_test = self.process_for_model_input(test)
            self.x_val, self.y_val = self.process_for_model_input(val)

    def sortData(self, data):
        ''' sort the genomic and clinical data w.r.t. survival time (OS_MONTHS) in descending order
        Input:
            path: path to input dataset (which is expected to be a csv file).
        Output:
            x: sorted genomic inputs.
            ytime: sorted survival time (OS_MONTHS) corresponding to 'x'.
            yevent: sorted censoring status (OS_EVENT) corresponding to 'x', where 1 --> deceased; 0 --> censored.
            age: sorted age corresponding to 'x'.
        '''

        data.sort_values("phe_os_time", ascending=False, inplace=True)
        x = data.drop(["phe_os_time", "phe_os", "phe_cancer", "phe_drugType", "phe_id"], axis=1).values
        ytime = data.loc[:, ["phe_os_time"]].values
        yevent = data.loc[:, ["phe_os"]].values

        X = tf.constant(x, dtype=tf.float32)
        YTIME = tf.constant(ytime, dtype=tf.float32)
        YEVENT = tf.constant(yevent, dtype=tf.float32)

        return X, YTIME, YEVENT