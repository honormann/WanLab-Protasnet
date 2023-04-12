import numpy as np
import tensorflow as tf
from keras.regularizers import l2
from Layer import Diagonal, Attention, AttentionFlow
from keras.layers import Input, Dense, Activation, Dropout
from keras import Model
from Loss import neg_par_log_likelihood_loss

class Protasnet:

    def __init__(self, dataloader=None):
        self.dataloader = dataloader
        self.input_dim = len([item for item in dataloader.X.columns if not 'phe' in item])
        self.sample = len(dataloader.X.index)
        self.omic = dataloader.config.cell_line_omic
        self.gene_graph = dataloader.adj
        self.model_args()

    def model_args(self):

        model_args = {
            "inputdim": self.input_dim,
            "omic_num": len(self.omic.split('_')),
            "activation": "tanh",
            'out_activation': 'sigmoid',
            "kernel_initializer": None,
            "use_bias": True,
            "w_regs": 0.01,
            "dropout": [0.25, 0.25],
            "learning_rate": 0.001,
            "epoch": 100,
            "epoch_lr_drop": True,
            "reduce_lr_drop": 0.25,
            "early_stopping": False,
            'batch_size': int(np.floor(self.sample * 0.8 / 24)),
            'loss_weight': [np.exp(0), np.exp(3)],
        }

        if self.dataloader.datatype == 'cell_line':
            model_args['num_nodes']=int(self.input_dim / len(self.omic.split('_')))
        else:
            model_args['num_nodes']=int(self.input_dim)

        # if self.dataloader.datatype == 'bulk':
        #     model_args['loss_weight'] = [np.exp(0), np.exp(0)]

        self.args = model_args

    def build_model(self):
        reg_l = l2
        inputs = Input(self.args["inputdim"])
        layer1 = Diagonal(self.args['num_nodes'], input_shape=(self.args["inputdim"],),
                          activation=self.args["activation"],
                          W_regularizer=reg_l(self.args["w_regs"]),
                          use_bias=self.args["use_bias"], name='h0',
                          kernel_initializer=self.args["kernel_initializer"])

        outcome = layer1(inputs)
        decision_outcomes = []
        decision_outcome = Dense(1, activation='linear', name='o_linear_node',
                                 kernel_regularizer=reg_l(self.args["w_regs"] / 2.))(outcome)
        if not self.dataloader.datatype == 'bulk':
            decision_outcome = Activation(activation=self.args["out_activation"], name='o_node')(decision_outcome)
        decision_outcomes.append(decision_outcome)
        drop0 = Dropout(self.args["dropout"][0], name='dropout_node')
        outcome = drop0(outcome, training=False)

        # attention
        att_layer = Attention(self.args['num_nodes'], name='attention_layer')
        gene_graph = tf.constant(self.dataloader.adj, dtype=tf.float32)
        a = att_layer([outcome, gene_graph])
        drop1 = Dropout(self.args['dropout'][1], name='dropout_att')
        a = drop1(a)
        att_flow = AttentionFlow(units=self.args['num_nodes'],
                                 activation=self.args["activation"],
                                 W_regularizer=l2(l2=self.args["w_regs"]),
                                 name='attenFlow',
                                 kernel_initializer=self.args["kernel_initializer"],
                                 use_bias=self.args["use_bias"])

        h = att_flow([outcome, a])
        decision_outcome = Dense(1, activation='linear', name='o_linear_com',
                                 kernel_regularizer=reg_l(self.args["w_regs"]))(h)
        if not self.dataloader.datatype == 'bulk':
            decision_outcome = Activation(activation=self.args["out_activation"], name='o_com')(decision_outcome)
        decision_outcomes.append(decision_outcome)

        self.protasnet = Model(inputs, decision_outcomes)

    def compile_model(self, type=None):

        if type == 'class':
            self.protasnet.compile(optimizer=tf.keras.optimizers.Adam(self.args["learning_rate"]),
                                  loss=[tf.keras.losses.binary_crossentropy] * 2,
                                  metrics=tf.keras.metrics.binary_accuracy,
                                  loss_weights=self.args['loss_weight']
                                      )

        if type == 'cox':
            self.protasnet.compile(optimizer=tf.keras.optimizers.Adam(self.args["learning_rate"]),
                                  loss=[neg_par_log_likelihood_loss] * 2,
                                  loss_weights=self.args['loss_weight']
                                      )
