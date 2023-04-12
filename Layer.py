import numpy as np

import tensorflow as tf
import keras.backend as K
from keras.layers import Layer
from keras import activations, initializers, constraints, regularizers


class Diagonal(Layer):
    def __init__(self, units, activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 W_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.units = units
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.W_regularizer = W_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = bias_constraint
        super(Diagonal, self).__init__(**kwargs)

    # the number of weights, equal the number of inputs to the layer
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dimension = input_shape[1]
        self.kernel_shape = (input_dimension, self.units)
        print ('input dimension {} self.units {}'.format(input_dimension, self.units))
        self.n_inputs_per_node = input_dimension / self.units
        print ('n_inputs_per_node {}'.format(self.n_inputs_per_node))

        rows = np.arange(input_dimension)
        cols = np.arange(self.units)
        cols = np.repeat(cols, self.n_inputs_per_node)
        self.nonzero_ind = np.column_stack((rows, cols))

        print ('self.kernel_initializer', self.W_regularizer, self.kernel_initializer, self.kernel_regularizer)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dimension,),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True, constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(Diagonal, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        n_features = tf.constant(x.shape[1], dtype=tf.int32)
        # print ('input dimensions {}'.format(x.shape))

        kernel = K.reshape(self.kernel, (1, n_features))
        mult = x * kernel
        mult = K.reshape(mult, (-1, tf.constant(int(self.n_inputs_per_node), dtype=tf.int32)))
        mult = K.sum(mult, axis=1)
        output = K.reshape(mult, (-1, tf.constant(self.units, dtype=tf.int32)))

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
        }
        base_config = super(Diagonal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PPiAttention(Layer):
    def __init__(self, units=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 W_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.W_regularizer = W_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = bias_constraint

        super(PPiAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.units,),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True, constraint=self.kernel_constraint)

        # self.kernel_attention = self.add_weight(
        #     shape=(2, 1),
        #     trainable=True,
        #     initializer=None,
        #     regularizer=None,
        #     name="kernel_attention",
        # )

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(PPiAttention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, data):

        feature, ppi = data

        # Linearly transform node states
        # features = data * self.kernel
        #
        # if self.use_bias:
        #     features = K.bias_add(features, self.bias)

        # Compute pair-wise attention scores
        h = tf.gather(feature, ppi, axis=1)
        h = tf.abs(h)
        h = tf.reduce_sum(h, axis=-1)
        h = h * self.kernel
        # h = tf.nn.leaky_relu(tf.matmul(h, self.kernel_attention))
        #
        # h = tf.reshape(h, shape=(-1, h.shape[1]))
        if self.use_bias:
            h = K.bias_add(h, self.bias)

        if self.activation_fn is not None:
            h = self.activation_fn(h)

        return h

    # def _call(self, data):
    #
    #     feature, ppi = data
    #
    #     # Linearly transform node states
    #     features = feature * self.kernel
    #
    #     # Compute pair-wise attention scores
    #     h = tf.gather(features, ppi, axis=1)
    #
    #     if self.use_bias:
    #         h = K.bias_add(h, self.bias)
    #
    #     h = tf.nn.leaky_relu(tf.matmul(h, self.kernel_attention))
    #
    #     h = tf.reshape(h, shape=(-1, h.shape[1]))
    #
    #     if self.activation_fn is not None:
    #         h = self.activation_fn(h)
    #
    #     return h

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0])

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
        }
        base_config = super(PPiAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class SparseTF(Layer):
    def __init__(self, units, map=None, nonzero_ind=None, kernel_initializer='glorot_uniform', W_regularizer=None,
                 activation='tanh', use_bias=True,
                 bias_initializer='zeros', bias_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.map = map
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activation_fn = activations.get(activation)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        super(SparseTF, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        # random sparse constarints on the weights
        # if self.map is None:
        #     mapp = np.random.rand(input_dim, self.units)
        #     mapp = mapp > 0.9
        #     mapp = mapp.astype(np.float32)
        #     self.map = mapp
        # else:
        if not self.map is None:
            self.map = self.map.astype(np.float32)

        # can be initialized directly from (map) or using a loaded nonzero_ind (useful for cloning models or create from config)
        if self.nonzero_ind is None:
            nonzero_ind = np.array(np.nonzero(self.map)).T
            self.nonzero_ind = nonzero_ind

        self.kernel_shape = (input_dim, self.units)
        # sA = sparse.csr_matrix(self.map)
        # self.sA=sA.astype(np.float32)
        # self.kernel_sparse = tf.SparseTensor(self.nonzero_ind, sA.data, sA.shape)

        # self.kernel_shape = (input_dim, self.units)
        # sA = sparse.csr_matrix(self.map)
        # self.sA=sA.astype(np.float32)
        # self.kernel_sparse = tf.SparseTensor(self.nonzero_ind, sA.data, sA.shape)
        # self.kernel_dense = tf.Variable(self.map)

        nonzero_count = self.nonzero_ind.shape[0]

        # initializer = initializers.get('uniform')
        # print 'nonzero_count', nonzero_count
        # self.kernel_vector = K.variable(initializer((nonzero_count,)), dtype=K.floatx(), name='kernel' )

        self.kernel_vector = self.add_weight(name='kernel_vector',
                                             shape=(nonzero_count,),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True, constraint=self.kernel_constraint)
        # self.kernel = tf.scatter_nd(self.nonzero_ind, self.kernel_vector, self.kernel_shape, name='kernel')
        # --------
        # init = np.random.rand(input_shape[1], self.units).astype( np.float32)
        # sA = sparse.csr_matrix(init)
        # self.kernel = K.variable(sA, dtype=K.floatx(), name= 'kernel',)
        # self.kernel_vector = K.variable(init, dtype=K.floatx(), name= 'kernel',)

        # print self.kernel.values
        # ind = np.array(np.nonzero(init))
        # stf = tf.SparseTensor(ind.T, sA.data, sA.shape)
        # print stf.dtype
        # print init.shape
        # # self.kernel = stf
        # self.kernel = tf.keras.backend.variable(stf, dtype='SparseTensor', name='kernel')
        # print self.kernel.values

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        super(SparseTF, self).build(input_shape)  # Be sure to call this at the end
        # self.trainable_weights = [self.kernel_vector]

    def call(self, inputs):
        # print self.kernel_vector.shape, inputs.shape
        # print self.kernel_shape, self.kernel_vector
        # print self.nonzero_ind
        # kernel_sparse= tf.S parseTensor(self.nonzero_ind, self.kernel_vector, self.kernel_shape)
        # pr = cProfile.Profile()
        # pr.enable()

        # print self.kernel_vector
        # self.kernel_sparse._values = self.kernel_vector
        tt = tf.scatter_nd(self.nonzero_ind, self.kernel_vector, self.kernel_shape)
        # print tt
        # update  = self.kernel_vector
        # tt= tf.scatter_add(self.kernel_dense, self.nonzero_ind, update)
        # tt= self.kernel_dense
        # tt[self.nonzero_ind].assign( self.kernel_vector)
        # self.kernel_dense[self.nonzero_ind] = self.kernel_vector
        # tt= tf.sparse.transpose(self.kernel_sparse)
        # output = tf.sparse.matmul(tt, tf.transpose(inputs ))
        # output = tf.matmul(tt, inputs )
        output = K.dot(inputs, tt)
        # pr.disable()
        # pr.print_stats(sort="time")
        # return tf.transpose(output)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            # 'kernel_shape': self.kernel_shape,
            'use_bias': self.use_bias,
            'nonzero_ind': np.array(self.nonzero_ind),
            # 'kernel_initializer': initializers.serialize(self.kernel_initializer),
            # 'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),

            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'W_regularizer': regularizers.serialize(self.kernel_regularizer),

        }
        base_config = super(SparseTF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # def call(self, inputs):
    #     print self.kernel.shape, inputs.shape
    #     tt= tf.sparse.transpose(self.kernel)
    #     output = tf.sparse.matmul(tt, tf.transpose(inputs ))
    #     return tf.transpose(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)



class Attention(Layer):
    def __init__(self, units, kernel_initializer='glorot_uniform',
                 **kwargs):
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.scale = tf.sqrt(tf.constant(units, dtype=tf.float32))

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):


        self.att_weight = self.add_weight(name='kernel_vector',
                                          shape=(self.units, self.units),
                                          initializer=self.kernel_initializer,
                                          trainable=True)


        super(Attention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):

        h, adj = inputs
        # adj = tf.constant(adj, dtype=tf.float32)
        energy = K.dot(tf.transpose(h), h)
        energy = K.dot(energy, self.att_weight)
        energy = energy/self.scale
        attention = tf.keras.activations.softmax(energy)
        attention = tf.multiply(attention, adj)

        return attention


    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),

        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def compute_output_shape(self, input_shape):
        return (self.units, self.units)



class AttentionFlow(Layer):
    def __init__(self, units, kernel_initializer='glorot_uniform', W_regularizer=None,
                 bias_regularizer=None,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_regularizer = bias_regularizer
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = bias_constraint
        super(AttentionFlow, self).__init__(**kwargs)

    def build(self, input_shape):

        self.kernel_vector = self.add_weight(name='kernel_vector',
                                             shape=(self.units, self.units),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True, constraint=self.kernel_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)

        super(AttentionFlow, self).build(input_shape)  # Be sure to call this at the end
        # self.trainable_weights = [self.kernel_vector]

    def call(self, inputs):
        h, a = inputs
        h = K.dot(h, self.kernel_vector)
        output = K.dot(h, a)

        if self.use_bias:
            output = K.bias_add(output, self.bias)

        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
        }
        base_config = super(AttentionFlow, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    def compute_output_shape(self, input_shape):
        return input_shape
