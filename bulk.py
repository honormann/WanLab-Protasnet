# -*- coding: utf-8 -*-

from Config import Config
from DataLoader import DataLoader
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load config
config = Config(graph_path="_data/string_interactions_short.tsv",
                graph_mapping="_data/string_mapping.tsv",
                cell_line_omic="mut_gain_loss_sv_exp",
                cell_line_fusion='exp',
                sc_cancer='lung')

# load data
dataloader = DataLoader(config=config)

## graph data
dataloader.loadGarph()

## bulk data
cancer = 'Melanoma'
dataloader.loadData(type='bulk')
dataloader.preprocess(cancer=cancer)  # Melanoma, Bladder Cancer

# build model: classification, cox
from Model import Protasnet
net = Protasnet(dataloader)
net.build_model()

# set random seed
import random
random.seed(123)
random_state = random.sample(range(1, 100), 30)

# split data
dataloader.splitData(random_state=random_state[1])

# train model
from Train import Trainer
trainer = Trainer(net=net, dataLoader=dataloader)

# set cv params
batch_size = [24, 36, 48]
# batch_size = [24]
learning_rate = [0.0005, 0.00035]
epoch = [48, 64, 96]
# epoch = [48]
l2 = [0.01]
droupout = [[0.25, 0.25]]

trainer.CV_params(batch_size=batch_size,
                  learning_rate=learning_rate,
                  epoch=epoch,
                  l2=l2,
                  droupout=droupout)

# grid search for best param
trainer.CV_search()

# train with best param
trainer.train_with_best_param()

# evaluate model
trainer.evaluate()

# log nodes value
import tensorflow as tf
import pandas as pd
exp = pd.read_csv('_data/bulk_external/' + "exp.csv")
id = exp.iloc[:, 0].tolist()
X = exp.drop(exp.columns[[0]], axis=1)

# X = dataloader.X
# X = X.loc[:, [items for items in X.columns if not 'phe' in items]]
h = tf.constant(X, dtype=tf.float32)
nodes_values = []
h1 = trainer.net.protasnet.layers[0](h)
h2 = trainer.net.protasnet.layers[1](h1)
h3 = trainer.net.protasnet.layers[2](h2)
a = trainer.net.protasnet.layers[3]([h3, dataloader.adj])
a = trainer.net.protasnet.layers[4](a)
h6 = trainer.net.protasnet.layers[5]([h3,a])
nodes = pd.DataFrame(h6.numpy())
nodes.columns = X.columns
nodes.index = id

dir_out = 'log/bulk_{}_explain/'.format(cancer)

nodes.to_csv(dir_out + 'nodes.csv')
ans = trainer.net.protasnet(h)
ans_trans = [pd.DataFrame(item.numpy()) for item in ans]
ans_trans[0].to_csv("log/external_1.csv")
ans_trans[1].to_csv("log/external_2.csv")

# explain model
from Explainer import Explainer
layer_name = ['h0', 'attention_layer', 'attenFlow']
explianer = Explainer(trainer=trainer,
                      dataLoader=dataloader,
                      layer_name=layer_name)
explianer.get_weights()
explianer.get_gradient_importance()
explianer.ann_result()


# log result
import csv
log = []
log.append(["batch_size: {}, learning_rate: {}, epoch: {}, l2: {}, droupout: {}".
                   format(trainer.best_param[0], trainer.best_param[1],
                          trainer.best_param[2], trainer.best_param[3],
                          trainer.best_param[4])])
log.append(["cIndex:{:.4f}, cIndex_ipcw:{:.4f}, brier_socre:{:.4f}".
           format(trainer.cIndex, trainer.cIndex_ipcw, trainer.brier_score)])

with open("./log/protasnet_bulk_{}.csv".format(cancer), "w",
          newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(log)

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

for i, item in enumerate(explianer.ann_grad):
    item.to_csv(dir_out + 'grad_{}.csv'.format(i), index=False)

for i, item in enumerate(explianer.ann_weight):
    item.to_csv(dir_out + 'weight_{}.csv'.format(i), index=True)
