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

## scSeq data
config.sc_cancer = 'lung'
dataloader.config = config
dataloader.loadData(type='scSeq')
### preprocess data
dataloader.preprocess()

# build model: classification, cox
from Model import Protasnet
net = Protasnet(dataloader)

# set random seed
import random
random.seed(123)
random_state = random.sample(range(1, 100), 30)

# split data
dataloader.splitData(random_state=random_state[0])

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
X = dataloader.X
X = X.loc[:, [items for items in X.columns if not 'phe' in items]]
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
nodes.index = dataloader.id

dir_out = 'log/sc_{}_explain/'.format(config.sc_cancer)

nodes.to_csv(dir_out + 'nodes.csv')

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
log.append(["acc:{:.4f}, auc:{:.4f}, f1:{:.4f}, recall:{:.4f}".format(trainer.acc, trainer.auc, trainer.f1_score, trainer.recall)])

with open("./log/protasnet_sc_{}.csv".format(config.sc_cancer), "w",
          newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(log)

if not os.path.exists(dir_out):
    os.makedirs(dir_out)

for i, item in enumerate(explianer.ann_grad):
    item.to_csv(dir_out + 'grad_{}.csv'.format(i), index=False)

for i, item in enumerate(explianer.ann_weight):
    item.to_csv(dir_out + 'weight_{}.csv'.format(i), index=True)
