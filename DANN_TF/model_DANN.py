from utils import *

class DANN(object):

    def __init__(self,embedding_size,sequence_length,label_classes,domain_classes,
                 filter_height,filter_width,num_filters_1,num_filters_2,hidden_dim, sup):

        self.X = tf.placeholder(tf.float32, [None, embedding_size, sequence_length, 1])
        self.y = tf.placeholder(tf.float32, [None, label_classes])
        self.domain = tf.placeholder(tf.int32, [None, domain_classes])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool)
        self.flip_gradient = FlipGradientBuilder()

        X_input=self.X
        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            W_conv0 = weight_variable([filter_height, filter_width,1, num_filters_1])
            b_conv0 = bias_variable([num_filters_1])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)

            #ksize: [batch,height,wide,in_channel], generally batch=1,channel=1
            # strides :[batch,height,wide,in_channel],generally batch=1,channel=1
            h_pool0=tf.nn.max_pool(h_conv0, ksize=[1, 1, 5, 1],
                           strides=[1, 1, 2, 1], padding='SAME')


            W_conv1 = weight_variable([filter_height, filter_width, num_filters_1, num_filters_2])
            b_conv1 = bias_variable([num_filters_2])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 1, 5, 1],
                           strides=[1, 1, 2, 1], padding='SAME')
            print("self.h_pool1+++++", h_pool1.shape)
            # The domain-invariant feature
            self.feature = tf.reshape(h_pool1, [-1, 4 * 26 * 48])

            print("self.feature+++++",self.feature.shape)
        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            # Switches to route target examples (second half of batch) differently
            # depending on train or test mode.
            all_features = self.feature
            print("all_featuresall_features",all_features.shape)
            print("self.trainvself.train",self.train)

            source_features = tf.slice(self.feature, [0, 0], [sup,-1])
            classify_feats = tf.cond(self.train,lambda: source_features, lambda:all_features)

            all_labels = self.y
            # source_labels=all_labels
            source_labels = tf.slice(self.y, [0, 0], [sup,-1])

            self.classify_labels = tf.cond(self.train, lambda:source_labels,lambda: all_labels)

            W_fc0 = weight_variable([4 * 26 * 48, hidden_dim])
            b_fc0 = bias_variable([hidden_dim])
            h_fc0 = tf.nn.relu(tf.matmul(classify_feats, W_fc0) + b_fc0)

            W_fc1 = weight_variable([hidden_dim, hidden_dim])
            b_fc1 = bias_variable([hidden_dim])
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)


            W_fc2 = weight_variable([hidden_dim, label_classes])
            b_fc2 = bias_variable([label_classes])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            print("self.logits+++++", logits.shape)
            print("self.self.classify_labels+++++", self.classify_labels.shape)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.classify_labels)

        # Small MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = self.flip_gradient(self.feature, self.l)

            d_W_fc0 = weight_variable([4 * 26 * 48, hidden_dim])
            d_b_fc0 = bias_variable([hidden_dim])
            d_h_fc0 = tf.nn.relu(tf.matmul(feat, d_W_fc0) + d_b_fc0)

            d_W_fc1 = weight_variable([hidden_dim, domain_classes])
            d_b_fc1 = bias_variable([domain_classes])
            d_logits = tf.matmul(d_h_fc0, d_W_fc1) + d_b_fc1
            print("self.d_logits+++++", d_logits.shape)
            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits, labels=self.domain)