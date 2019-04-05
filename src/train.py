import tensorflow as tf
import os

from data_processor import DataProcesser
from graph_constructor import CNNGraph

class TrainCNN(object):
    def __init__(self, image_size, input_channel,class_names, model_path,
                 conv_filter_size, conv_nums_filters, fc_layer_size, batch_size):
        self.model_path = model_path
        # Create the model path if needed
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.batch_size = batch_size

        self.input_x = image_size
        self.input_y = image_size
        self.input_channel = input_channel
        self.num_classes = len(class_names)
        self.conv_filter_size = conv_filter_size
        self.conv_nums_filters = conv_nums_filters
        self.fc_layer_size = fc_layer_size
        self.optimizer = None
        self.cost = None
        self.cross_entropy = None
        self.data_process = DataProcesser(class_names, image_size)
        self.graph_constructor = CNNGraph()

    def initialize_graph(self):
        self.graph_constructor.create_graph(self.input_x, self.input_y, self.input_channel, self.num_classes,
                                            self.conv_filter_size, self.conv_nums_filters, self.fc_layer_size)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.graph_constructor.last_layer,
                                                                     labels=self.graph_constructor.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.cost)
        self.correct_prediction = tf.equal(self.graph_constructor.y_pred_cls,
                                           self.graph_constructor.y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def read_total_iter(self):
        tt_iter_file = os.path.join(self.model_path, 'total_iteration.txt')
        total_iter = 0
        if not os.path.exists(tt_iter_file):
            return total_iter
        with open(tt_iter_file, 'r') as f:
            try:
                total_iter= int(f.read())
            except:
                print('Error reading the total number of iterations. Assigning zero')
        return total_iter

    def write_total_iter(self, total_iter):
        tt_iter_file = os.path.join(self.model_path, 'total_iteration.txt')
        with open(tt_iter_file, 'w+') as f:
            try:
                f.write(str(total_iter))
            except:
                raise('Error writing the total number of iterations. Assigning zero')

    def load_data(self, data_path, training_set_size):
        self.data_process.read_data_from_folder(data_path)
        self.data_process.divide_training_valid_data(training_set_size)
            
    def show_progress(self, epoch, feed_dict_train, feed_dict_validate, val_loss, session):
        acc = session.run(self.accuracy, feed_dict=feed_dict_train)
        val_acc = session.run(self.accuracy, feed_dict=feed_dict_validate)
        msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc, val_acc, val_loss))
                
    def train(self, num_iterations):
        ''' Train the network for certain number of iterations
        param:
        num_iterations: The number of iterations to train
        '''
        # TODO:: NEED TO LOAD AND SAVE THE TOTAL ITER
        total_iter_file = os.path.join(self.model_path, '/total_iter.txt')
        total_num_iter = 0
        with tf.Session() as session:
            self.initialize_graph()
            saver = tf.train.Saver()
            x = self.graph_constructor.x
            y_true = self.graph_constructor.y_true

            session.run(tf.global_variables_initializer())
            if os.path.isfile(self.model_path + 'trained_model-0.meta'):
                # Model has been trained. Restore
                saver.restore(session, tf.train.latest_checkpoint(self.model_path))
                total_num_iter = self.read_total_iter()
            else:
                total_num_iter = self.read_total_iter()

            for step in range(total_num_iter, total_num_iter + num_iterations):
                x_batch_train, y_batch_train = self.data_process.next_training_batch(self.batch_size)
                x_batch_valid, y_batch_valid = self.data_process.next_validate_batch(self.batch_size)
                feed_dict_train = {x: x_batch_train,
                                   y_true: y_batch_train}
                feed_dict_valid = {x: x_batch_valid,
                                   y_true: y_batch_valid}
                session.run(self.optimizer, feed_dict=feed_dict_train)
                if step % int(self.data_process.get_training_data_size()/self.batch_size) == 0: 
                    val_loss = session.run(self.cost, feed_dict=feed_dict_valid)
                    epoch = int(step / int(self.data_process.get_training_data_size()/self.batch_size))
            
                    self.show_progress(epoch, feed_dict_train, feed_dict_valid, val_loss, session)
                    saver.save(session, self.model_path + 'trained_model', global_step=step) 
                    total_num_iter = step
                    self.write_total_iter(total_num_iter)
