import sys
from Model import Protasnet
import numpy as np
import pandas as pd
import tensorflow as tf
from DeepExplain import DeepExplain


class Explainer:

    def __init__(self, trainer=None, dataLoader=None, layer_name=None):
        self.trainer = trainer
        self.net = trainer.net.protasnet
        self.dataLoader = dataLoader
        self.layer_name = layer_name

    def get_weights(self):
        weights = []
        for item in self.layer_name:

            sub_layer = self.net.get_layer(item)
            sub_w = sub_layer.get_weights()[0]
            weights.append(sub_w)

        self.weights = weights

    def get_gradient_importance(self):

        ww = self.net.get_weights()

        output_index = -1
        method_name = 'deeplift'
        with tf.compat.v1.Session() as sess:
            try:
                with DeepExplain(session=sess) as de:  # <-- init DeepExplain context

                    print(self.layer_name)
                    net = Protasnet(dataloader=self.dataLoader)
                    batch, lr, epoch, l2, droupout = self.trainer.best_param
                    net.args["batch_size"] = batch
                    net.args["learning_rate"] = lr
                    net.args["epoch"] = epoch
                    net.args["l2"] = l2
                    net.args["dropout"] = droupout

                    net.build_model()
                    # net.compile_model(type='class')
                    model = net.protasnet
                    model.set_weights(ww)

                    gradient_score = {}
                    for sublayer in self.layer_name:
                        x = model.get_layer(sublayer).output
                        if type(output_index) == str:
                            y = model.get_layer(output_index).output
                        else:
                            y = model.outputs[output_index]


                        print(self.layer_name)
                        print('model.inputs', model.inputs)
                        print('model y', y)
                        print('model x', x)
                        attributions = de.explain(method_name, y, x, model.inputs[0], self.dataLoader.x_train.numpy())
                        print('attributions', attributions.shape)
                        gradient_score[sublayer] = attributions

                    self.gradient_score = gradient_score

            except:
                sess.close()
                print("Unexpected error:", sys.exc_info()[0])
                raise

    def ann_result(self):

        ann_grad = []
        gene_grad = {}
        gene_grad['grad'] = np.sum(self.gradient_score['h0'], axis=0)
        gene_grad['gene'] = self.dataLoader.gene_dict.keys()
        gene_grad = pd.DataFrame(gene_grad)
        ann_grad.append(gene_grad)

        attention_grad = self.gradient_score['attention_layer']
        attention_grad = pd.DataFrame(attention_grad)
        attention_grad.columns = self.dataLoader.gene_dict.keys()
        attention_grad.index = self.dataLoader.gene_dict.keys()
        ann_grad.append(attention_grad)

        com_grad = {}
        com_grad['grad'] = np.sum(self.gradient_score['attenFlow'], axis=0)
        com_grad['gene'] = self.dataLoader.gene_dict.keys()
        com_grad = pd.DataFrame(com_grad)
        ann_grad.append(com_grad)

        ann_weight = []
        for id, t in enumerate(self.weights):
            if id == 0:
                result = {}
                result['weight'] = self.weights[id]
                result['genes'] = [i for i in self.dataLoader.X.columns if not "phe" in i]
                result = pd.DataFrame(result)
            else:
                result = pd.DataFrame(self.weights[id])
                result.index = self.dataLoader.gene_dict.keys()
                result.columns = self.dataLoader.gene_dict.keys()

            ann_weight.append(result)

        self.ann_grad = ann_grad
        self.ann_weight = ann_weight
