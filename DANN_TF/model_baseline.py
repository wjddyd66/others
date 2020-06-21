from utils import *

class DANN(object):
    def __init__(self, batch_size):
        self.X = tf.placeholder(tf.float32, [None, 4, 101, 1])
        self.y = tf.placeholder(tf.float32, [None, 2])

        X_input = self.X

        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([4, 5, 1, 32])
            b_conv0 = bias_variable([32])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = tf.nn.max_pool(h_conv0, ksize=[1, 1, 5, 1],
                                     strides=[1, 1, 2, 1], padding='SAME')
            # h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([4, 5, 32, 48])
            b_conv1 = bias_variable([48])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 5, 1],
                                     strides=[1, 1, 2, 1], padding='SAME')
            print("self.h_pool1+++++", h_pool1.shape)
            # The domain-invariant feature
            self.feature = tf.reshape(h_pool1, [-1, 4 * 26 * 48])

            print("self.feature+++++", self.feature.shape)

        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = self.feature
            classify_feats =  all_features

            all_labels = self.y


            self.classify_labels = all_labels

            W_fc0 = weight_variable([4 * 26 * 48, 100])
            b_fc0 = bias_variable([100])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc1 = weight_variable([100, 100])
            b_fc1 = bias_variable([100])
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([100, 2])
            b_fc2 = bias_variable([2])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)