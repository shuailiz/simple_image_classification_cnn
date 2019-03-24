import tf
import os

from data_processor import DataProcesser
from graph_constructor import CNNGraph

class TrainCNN(object):
    def __init__(self, input_x, input_y, input_channel,
                 num_classes, conv_filter_size, conv_nums_filters, fc_layer_size):
        self.model_path = os.path.join(os.getcwd(), '../models/')
        self.batch_size = 50

        self.input_x = input_x
        self.input_y = input_y
        self.input_channel = input_channel
        self.num_classes = num_classes
        self.conv_filter_size = conv_filter_size
        self.conv_nums_filters = conv_nums_filters
        self.fc_layer_size = fc_layer_size
        self.optimizer = None
        self.cost = None
        self.cross_entropy = None
        self.data_process = DataProcesser()
        self.data_process.divide_training_valid_data(0.85)
        self.graph_constructor = CNNgraph()

    def initialize_graph(self):
        self.graph_constructor.create_graph(self.input_x, self.input_y, self.input_channel, self.num_classes,
                                            self.conv_filter_size, self.nums_filters, self.fc_layer_size)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.graph_constructor.last_layer,
                                                                     labels=self.graph_constructor.y_true)
        self.cost = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        self.correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def read_total_iter(self):
        tt_iter_file = os.path.join(self.model_dir, 'total_iteration.txt')
        with open(tt_iter_file, 'r') as f:
            try:
                self.total_iter = int(f.read())
            except:
                print('Error reading the total number of iterations. Assigning zero')
                self.total_iter = 0

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
            if os.path.isfile(self.model_path + '.meta'):
                # Model has been trained. Restore
                saver = tf.import_meta_graph(self.model_path + ".meta")
                saver.restore(session, tf.train.latest_checkpoint(self.model_path))
                with open(total_iter_file, 'r') as total_iter_file:
                    total_num_iter = int(total_iter_file.read())
            else:
                # Otherwise create the graph
                self.initialize_graph()
                self.read_total_iter()
                saver = tf.train.Saver()

            session.run(tf.global_variables_initializer())
            for step in range(total_num_iter, total_num_iter + num_iterations):
                x_batch_train, y_batch_train = self.data_process.next_training_batch(self.batch_size)
                x_batch_valid, y_batch_valid = self.data_process.next_validate_batch(self.batch_size)
                feed_dict_train = {x: x_batch_train
                                   y_true: y_batch_train}
                feed_dict_valid = {x: x_batch_valid,
                                   y_true: y_batch_valid}
                session.run(self.optimizer, feed_dict=feed_dict_train)
                if step % int(self.data_process.get_training_data_size()/batch_size) == 0: 
                    val_loss = session.run(self.cost, feed_dict=feed_dict_valid)
                    epoch = int(step / int(self.data_process.get_training_data_size()/batch_size))
            
                    show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, session)
                    saver.save(session, self.model_path + 'trained_model', global_step=step) 
                    total_num_iter = step

            with open(total_iter_file, 'w') as total_iter_file:
                total_iter_file.write(total_num_iter)
