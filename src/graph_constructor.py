#!/usr/bin/python3
import tensorflow as tf
import copy

class CNNGraph(object):
    ''' Create a tf graph for converlutional nural network'''

    def __init__(self):
        pass

    def reset(self):
        self.conv_layers = []
        self.fc_layers = []
        self.last_layer = None
        self.x = None
        self.y_true = None
        self.y_true_cls = None
        self.y_pred = None
        self.y_pred_cls = None

    def create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

    def create_biases(self, size):
        return tf.Variable(tf.constant(0.05, shape=[size]))

    def create_convolution_layer(self, input_data,
                                 num_input_channels,
                                 conv_filter_size,
                                 num_filters):
        weights = self.create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
        biases = self.create_biases(size=num_filters)


        x_stride = 1
        y_stride = 1

        # Conv layer
        layer = tf.nn.conv2d(input=input_data,
                             filter=weights,
                             strides=[1, x_stride, y_stride, 1],
                             padding='SAME')

        layer += biases

        # Max pooling layer
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

        # Relu layer
        layer = tf.nn.relu(layer)

        return layer

    def create_flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, num_features])
        return layer

    def create_fc_layer(self, input_layer,
                        input_dim,
                        output_dim,
                        use_relu=True):
        weights = self.create_weights(shape=[input_dim, output_dim])
        biases = self.create_biases(size=output_dim)

        layer = tf.matmul(input_layer, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer

    def create_graph(self, input_size_x, input_size_y, input_channels, num_classes, conv_filter_sizes, conv_nums_filters, fc_layer_sizes):
        self.conv_layers = []
        self.fc_layers = []
        self.x = tf.placeholder(tf.float32, shape=[None, input_size_x, input_size_y, input_channels], name='x')
        self.y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
        self.y_true_cls = tf.argmax(self.y_true, dimension=1)
        layer_input = self.x
        layer_input_channels = input_channels
        # create all the conv layers
        for conv_filter_size, conv_num_filters in zip(conv_filter_sizes, conv_nums_filters):
            conv_layer = self.create_convolution_layer(input_data=layer_input,
                                                       num_input_channels=layer_input_channels,
                                                       conv_filter_size=conv_filter_size,
                                                       num_filters=conv_num_filters)
            layer_input = conv_layer
            layer_input_channels = conv_num_filters
            self.conv_layers.append(conv_layer)

        print(self.conv_layers)
        # create the flat layer
        self.layer_flat = self.create_flatten_layer(self.conv_layers[-1])

        # create the fc layers
        layer_input = self.layer_flat
        layer_input_dim = self.layer_flat.get_shape()[1:4].num_elements()
        for fc_layer_size in fc_layer_sizes:
            fc_layer = self.create_fc_layer(input_layer=layer_input,
                                            input_dim=layer_input_dim,
                                            output_dim=fc_layer_size,
                                            use_relu=True)
            layer_input = fc_layer
            layer_input_dim = fc_layer_size
            self.fc_layers.append(fc_layer)

        # create the last layer for class prediction
        self.last_layer = self.create_fc_layer(input_layer=self.fc_layers[-1],
                                               input_dim=fc_layer_sizes[-1],
                                               output_dim=num_classes,
                                               use_relu=False)
        self.y_pred = tf.nn.softmax(self.last_layer, name='y_pred')
        self.y_pred_cls = tf.argmax(self.y_pred, dimension=1)
