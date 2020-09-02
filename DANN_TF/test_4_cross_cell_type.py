import tensorflow as tf
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from utils import *
import sys
import re
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import gzip
from model_DANN import DANN
import random

def clean_data(string):
    string = re.sub(r"[A]", "0", string)
    string = re.sub(r"[C]", "1", string)
    string = re.sub(r"[G]", "2", string)
    string = re.sub(r"[T]", "3", string)
    return string


def load_file(filename_pos, filename_neg):
    positive_text_x = []
    negative_text_x = []

    positive_examples = open(filename_pos, "r").readlines()
    positive_examples = [s.strip() for s in positive_examples]
    for i in range(positive_examples.__len__()):
        if (i % 2 == 1):
            positive_text_x.append(positive_examples[i])
    positive_labels = [[0., 1.] for s in positive_text_x]
    # print(positive_text_x.__len__())

    negative_examples = open(filename_neg, "r").readlines()
    negative_examples = [s.strip() for s in negative_examples]
    for i in range(negative_examples.__len__()):
        if (i % 2 == 1):
            negative_text_x.append(negative_examples[i])
    negative_labels = [[1., 0.] for s in negative_text_x]

    positive_text_x.extend(negative_text_x)

    x_text = [clean_data(s) for s in positive_text_x]
    y_labels = np.concatenate((positive_labels, negative_labels), axis=0)

    return x_text, y_labels


def load_data_and_labels(target_cell, target_tf):
    print('Target Cell: {}, Target TF: {}'.format(target_cell, target_tf))
    source_cell = ["GM12878", "K562", "HeLa-S3", "HepG2", "H1-hESC"]
    source_cell.remove(target_cell)

    x_CTCF, y_CTCF = load_file("./data/" + source_cell[0] + "/" + target_tf + ".fasta",
                               "./data/" + source_cell[0] + "/" + target_tf + "_back.fasta")

    domain_CTCF = np.tile([0, 0, 0, 0, 1], [len(x_CTCF), 1])

    x_GABP, y_GABP = load_file("./data/" + source_cell[1] + "/" + target_tf + ".fasta",
                               "./data/" + source_cell[1] + "/" + target_tf + "_back.fasta")
    domain_GABP = np.tile([0, 0, 0, 1, 0], [len(x_GABP), 1])

    x_JunD, y_JunD = load_file("./data/" + source_cell[2] + "/" + target_tf + ".fasta",
                               "./data/" + source_cell[2] + "/" + target_tf + "_back.fasta")
    domain_JunD = np.tile([0, 0, 1, 0, 0], [len(x_JunD), 1])

    x_REST, y_REST = load_file("./data/" + source_cell[3] + "/" + target_tf + ".fasta",
                               "./data/" + source_cell[3] + "/" + target_tf + "_back.fasta")
    domain_REST = np.tile([0, 1, 0, 0, 0], [len(x_REST), 1])

    # target data
    x_USF2, y_USF2 = load_file("./data/" + target_cell + "/" + target_tf + ".fasta",
                               "./data/" + target_cell + "/" + target_tf + "_back.fasta")

    x_CTCF.extend(x_GABP)
    x_CTCF.extend(x_JunD)
    x_CTCF.extend(x_REST)
    x_source = x_CTCF
    source_domain = np.vstack([domain_CTCF, domain_GABP, domain_JunD, domain_REST])
    y_source = np.concatenate((y_CTCF, y_GABP), axis=0)
    y_source = np.concatenate((y_source, y_JunD), axis=0)
    y_source = np.concatenate((y_source, y_REST), axis=0)
    x_target = x_USF2
    y_target = y_USF2

    return x_source, y_source, x_target, y_target, source_domain


def load_one_hot(data):
    tx = []
    enc = OneHotEncoder()
    enc.fit([[0] * 101, [1] * 101, [2] * 101, [3] * 101])
    for i in tqdm(range(data.__len__())):
        a = enc.transform([np.array(list(data[i]), dtype=int)]).toarray().reshape(101, 4)
        a = a.T
        tx.append(a)
    return tx


def train_shuffle_data(x_data, y_data, domain_data):
    print("Shuffle Train data...")

    np.random.seed(100)
    shuffle_indices = np.random.permutation(np.arange(len(y_data)))
    x_s = []
    y_s = []
    domain_s = []
    for i in range(len(y_data)):
        x_s.append(x_data[shuffle_indices[i]])
        y_s.append(y_data[shuffle_indices[i]])
        domain_s.append(domain_data[shuffle_indices[i]])
    return x_s, y_s, domain_s


def test_shuffle_data(x_data, y_data):
    print("Shuffle Test data...")
    np.random.seed(100)
    shuffle_indices = np.random.permutation(np.arange(len(y_data)))
    x_s = []
    y_s = []
    for i in range(len(y_data)):
        x_s.append(x_data[shuffle_indices[i]])
        y_s.append(y_data[shuffle_indices[i]])
    return x_s, y_s


def batch_iter(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    data_size = int(data_size / batch_size) * batch_size
    print("Data size ", data_size)
    num_batch_per_epochs = int((data_size - 1) / batch_size) + 1
    print("Num_batch_per_epochs", num_batch_per_epochs)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffle_data = data[shuffle_indices]
    else:
        shuffle_data = data

    for batch_num in range(num_batch_per_epochs):
        start_index = batch_num * batch_size
        # if((batch_num + 1) * batch_size == data_size)

        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffle_data[start_index:end_index]


def data_initialize(target_cell, target_tf):
    x_source, y_source, x_target, y_target, source_domain = load_data_and_labels(target_cell, target_tf)

    data_path = './data/Model/' + target_cell + '/' + target_tf + '/'
    x_source_file_path = data_path + 'x_source.pickle'
    y_source_file_path = data_path + 'y_source.pickle'
    x_target_file_path = data_path + 'x_target.pickle'
    y_target_file_path = data_path + 'y_target.pickle'
    domain_file_path = data_path + 'domain.pickle'

    if (os.path.isfile(x_source_file_path) and os.path.isfile(y_source_file_path)
            and os.path.isfile(x_target_file_path) and os.path.isfile(y_target_file_path)
            and os.path.isfile(domain_file_path)):

        print('Data Existing.... => Loading')

        # Load Source data
        with gzip.open(x_source_file_path, 'rb') as f:
            x_source = pickle.load(f)

        with gzip.open(y_source_file_path, 'rb') as f:
            y_source = pickle.load(f)

        with gzip.open(x_target_file_path, 'rb') as f:
            x_target = pickle.load(f)

        with gzip.open(y_target_file_path, 'rb') as f:
            y_target = pickle.load(f)

        with gzip.open(domain_file_path, 'rb') as f:
            source_domain = pickle.load(f)

        return x_source, y_source, x_target, y_target, source_domain

    else:
        # Source Data
        print("Loading source one hot...")
        x_source, y_source, domain_source = train_shuffle_data(x_source, y_source, source_domain)
        x_source = load_one_hot(x_source)

        # Save Source Data
        with gzip.open(x_source_file_path, 'wb') as sx:
            print('Size of x_source: ', sys.getsizeof(x_source))
            pickle.dump(x_source, sx)
        with gzip.open(y_source_file_path, 'wb') as sy:
            pickle.dump(y_source, sy)

        # Save Domain Data
        with gzip.open(domain_file_path, 'wb') as d:
            pickle.dump(source_domain, d)

        # Target Data
        print("Loading target one hot...")
        x_target, y_target = test_shuffle_data(x_target, y_target)
        x_target = load_one_hot(x_target)
        x_target = np.array(x_target)

        # Save Target Data
        with gzip.open(x_target_file_path, 'wb') as tx:
            print('Size of x_target: ', sys.getsizeof(x_target))
            pickle.dump(x_target, tx)
        with gzip.open(y_target_file_path, 'wb') as ty:
            pickle.dump(y_target, ty)

        print("Saving Data set Finished....")

        return x_source, y_source, x_target, y_target, source_domain

def train_model(target_cell, target_tf, supervise):
    sup_flag = 32

    flags = tf.app.flags
    # Data loading params
    flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

    # Model Hyperparameters
    flags.DEFINE_integer("embedding_size1", 4, "length of sequence embedding(default: 4)")
    flags.DEFINE_integer("sequence_length", 101, "length of sequence (default: 101)")

    flags.DEFINE_integer("filter_height", 4, "Comma-separated filter sizes ")
    flags.DEFINE_integer("filter_width", 5, "Comma-separated filter sizes ")
    flags.DEFINE_integer("num_filters_1", 32, "Number of filters per filter size")
    flags.DEFINE_integer("num_filters_2", 48, "Number of filters per filter size")
    flags.DEFINE_integer("hidden_dim", 100, "FC hidden dim")
    flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    flags.DEFINE_float("learning_rate", 1e-3, "L2 regularization lambda (default: 0.0)")
    #
    # Training parameters
    flags.DEFINE_integer("embedding_size", 4, "length of sequence embedding(default: 4)")
    flags.DEFINE_integer("label_classes", 2, "The number of categories for the label (default: 2)")
    flags.DEFINE_integer("domain_classes", 5, "The number of categories for the domain (default: 2)")
    flags.DEFINE_integer("dev_batch_size", 2048, "dev_Batch Size (default: 2048)")
    flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 200)")
    flags.DEFINE_integer("evaluate_every", 10000, "Evaluate model on dev set after this many steps (default: 100)")
    # tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")

    # Misc Parameters
    # log_device_placement=True
    # allow_soft_placement=True
    flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = flags.FLAGS
    # FLAGS._parse_flags()
    FLAGS(sys.argv)

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    #######################################################################################################################
    # Load data
    x_source, y_source, x_target, y_target, source_domain = data_initialize(target_cell, target_tf)

    target_len = y_target.__len__()
    print("x_source", x_source.__len__())
    print("y_target", y_target.__len__())

    x_source_train = np.expand_dims(np.array(x_source), axis=-1)
    y_source_train = np.array(y_source)

    ########################################################################################################################
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = "./checkpoints/cnn/" + target_tf + '_' + target_cell + '_' + str(supervise)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Build the model graph
    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True

        model = DANN(embedding_size=FLAGS.embedding_size1,
                     sequence_length=FLAGS.sequence_length,
                     label_classes=FLAGS.label_classes,
                     domain_classes=FLAGS.domain_classes,
                     filter_height=FLAGS.filter_height,
                     filter_width=FLAGS.filter_width,
                     num_filters_1=FLAGS.num_filters_1,
                     num_filters_2=FLAGS.num_filters_2,
                     hidden_dim=FLAGS.hidden_dim,
                     sup=sup_flag)
        learning_rate = FLAGS.learning_rate
        global_step = tf.Variable(0, name="global_step", trainable=False)
        pred_loss = tf.reduce_mean(model.pred_loss)
        domain_loss = tf.reduce_mean(model.domain_loss)
        total_loss = pred_loss + domain_loss

        regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(total_loss)

        # Evaluation
        correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
        label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
        domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))

        # ROC AUC
        AUC_label = tf.argmax(model.classify_labels, 1)
        pred_label = tf.argmax(model.pred, 1)
        AUC_score = tf.split(model.pred, num_or_size_splits=2, axis=1)

        def cross(x_target, y_target):
            AUROC_total = 0.0
            target_acc_total = 0.0
            f1_value_total = 0.0
            for i in range(10):
                sess = tf.Session(config=session_conf)
                with sess.as_default():

                    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                    sess.run(tf.global_variables_initializer())

                    def train_step(x_batch, y_batch, domain_labels, l):
                        _, batch_loss, dloss, ploss, d_acc, p_acc, auc_label = \
                            sess.run([dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc,
                                      AUC_label],
                                     feed_dict={model.X: x_batch, model.y: y_batch, model.domain: domain_labels,
                                                model.train: True, model.l: l
                                                })

                    def dev_step(x_batch_target, y_batch_target):
                        # """
                        # Evaluates model on a dev set
                        # """
                        acc, auc_label, auc_score = sess.run([label_acc, AUC_label, AUC_score],
                                                             feed_dict={model.X: x_batch_target,
                                                                        model.y: y_batch_target,
                                                                        model.train: False})
                        AUROC = roc_auc_score(auc_label, auc_score[1])
                        target_acc = acc
                        print("Evaluation " + str(i) + " :")
                        print(
                            'Evaluation: AUROC: %f  target_acc: %f ' % \
                            (AUROC, target_acc))

                        return AUROC, target_acc

                    def Test(p_x_target_test, p_y_target_test, step):
                        model_file = tf.train.latest_checkpoint(checkpoint_dir + "/" + str(step))
                        saver.restore(sess, model_file)

                        acc, auc_label, auc_score, f1_label = sess.run(
                            [label_acc, AUC_label, AUC_score, pred_label],
                            feed_dict={model.X: p_x_target_test,
                                       model.y: p_y_target_test,
                                       model.train: False})
                        AUROC = roc_auc_score(auc_label, auc_score[1])
                        data_label = pd.DataFrame(auc_label)
                        data_score = pd.DataFrame(auc_score[1])
                        data_label.to_csv('./auROC/cnn/label/' + target_tf + '_' + target_cell + '_' + str(
                            supervise) + '.csv', mode='a', header=False, index=False)
                        data_score.to_csv('./auROC/cnn/score/' + target_tf + '_' + target_cell + '_' + str(
                            supervise) + '.csv', mode='a', header=False, index=False)
                        target_acc = acc
                        f1 = f1_score(auc_label, f1_label)
                        file_object = open(
                            './cnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '', 'a')

                        strint = "Test  :" + "AUC: " + str(
                            AUROC) + "f1:" + str(f1) + "target_acc: " + str(target_acc) + "\n"

                        file_object.write(strint)
                        file_object.close()
                        return AUROC, f1, target_acc

                    def trainAndDev(p_x_source_train, p_y_source_train, p_x_target_train, p_y_target_train,
                                    p_x_lan_t,
                                    p_y_target_dev, step, p_domain_source):

                        best_target_AUROC = 0.5
                        checkpoint_prefix = os.path.join(checkpoint_dir + "/" + str(step), "model")
                        for epoch in range(FLAGS.num_epochs):
                            #     # Generate batches
                            print('\nbegin trainAndDev')
                            gen_source_batch = batch_iter(
                                list(zip(p_x_source_train, p_y_source_train, p_domain_source)),
                                int(FLAGS.batch_size / 2))
                            gen_target_batch = batch_iter(
                                list(zip(p_x_target_train, p_y_target_train)), int(FLAGS.batch_size / 2))

                            # Training step
                            X0 = []
                            y0 = []
                            X1 = []
                            y1 = []
                            domain0 = []
                            for batch in gen_source_batch:
                                X, y, domain = zip(*batch)

                                X0.append(X)
                                y0.append(y)
                                domain0.append(domain)
                            for batch in gen_target_batch:
                                X, y = zip(*batch)
                                X1.append(X)
                                y1.append(y)

                            X0 = np.array(X0)
                            X1 = np.array(X1)
                            y0 = np.array(y0)
                            y1 = np.array(y1)
                            domain0 = np.array(domain0)

                            x_i = []
                            y_i = []
                            domain_i = []

                            for i in range(X0.shape[0]):
                                x_temp = random.choice(range(X1.shape[0]))
                                x = np.concatenate((X0[i], X1[x_temp]), axis=0)
                                y_temp = np.concatenate((y0[i], y1[x_temp]), axis=0)
                                x_i.append(x)
                                y_i.append(y_temp)
                                domain_i.append((domain0[i]))

                            x_i = np.array(x_i)
                            y_i = np.array(y_i)
                            domain_i = np.array(domain_i)

                            print("Source Domain Shape", domain0.shape)
                            print("Source Data Shape", X0.shape)
                            print("Source Label Shape", y0.shape)

                            print("Target Label Shape", y1.shape)

                            file_object = open(
                                './cnn/dev/devValue_' + target_tf + '_' + target_cell + '_' + str(
                                    supervise) + '', 'a')

                            for j in range(x_i.shape[0]):
                                p = float(j) / X0.shape[0]
                                l = 2. / (1. + np.exp(-10. * p)) - 1

                                domain_labels = np.vstack([domain_i[j],
                                                           np.tile([1, 0, 0, 0, 0], [32, 1])])

                                train_step(x_i[j], y_i[j], domain_labels, l)
                                if j % FLAGS.evaluate_every == 0:
                                    print("\nEvaluation")

                                    target_AUROC, target_acc = dev_step(p_x_lan_t, p_y_target_dev)

                                    if best_target_AUROC < target_AUROC:
                                        best_target_AUROC = target_AUROC
                                        path = saver.save(sess, checkpoint_prefix,
                                                          global_step=(j + 1) * (epoch + 1))
                                        print("Saved model checkpoint to {}\n".format(path))

                                    strint = "ecoch :" + str(epoch) + "Evaluation  :" + str(
                                        j / FLAGS.evaluate_every) + "AUC: " + str(
                                        target_AUROC) + "target_acc: " + str(target_acc) + "\n"

                                    file_object.write(strint)

                                    print("")
                            file_object.close()



                    if i == 9:
                        x_target_test = x_target[int(target_len * 0.9):]
                        y_target_test = y_target[int(target_len * 0.9):]
                        x_target_dev = x_target[:int(target_len * 0.1)]
                        y_target_dev = y_target[:int(target_len * 0.1)]
                        x_target_train = x_target[int(target_len * 0.1):int(target_len * 0.9)]
                        y_target_train = y_target[int(target_len * 0.1):int(target_len * 0.9)]
                        x_target_train = np.expand_dims(np.array(x_target_train), axis=-1)
                        y_target_train = np.array(y_target_train)

                    else:
                        if i == 0:
                            test_start_index = 0
                        else:
                            test_start_index = int(target_len * (i / 10))


                        test_end_index = int(target_len * ((i + 1) / 10))
                        validation_end_index = int(target_len * ((i + 2) / 10))

                        x_target_test = x_target[test_start_index:test_end_index]
                        y_target_test = y_target[test_start_index:test_end_index]
                        x_target_dev = x_target[test_end_index:validation_end_index]
                        y_target_dev = y_target[test_end_index:validation_end_index]

                        if i == 0:
                            x_target_train = x_target[validation_end_index:]
                            y_target_train = y_target[validation_end_index:]

                        else:
                            x_target_train = np.concatenate(
                                (x_target[:test_start_index], x_target[validation_end_index:]))
                            y_target_train = np.concatenate(
                                (y_target[:test_start_index], y_target[validation_end_index:]))



                    x_target_train = np.expand_dims(np.array(x_target_train), axis=-1)
                    y_target_train = np.array(y_target_train)

                    x_target_dev = np.expand_dims(np.array(x_target_dev), axis=-1)
                    y_target_dev = np.array(y_target_dev)

                    x_target_test = np.expand_dims(np.array(x_target_test), axis=-1)
                    y_target_test = np.array(y_target_test)
                    x_lan_t = x_target_dev

                    if not os.path.exists(checkpoint_dir + "/" + str(i + 1)):
                        os.makedirs(checkpoint_dir + "/" + str(i + 1))

                    # Train
                    trainAndDev(x_source_train, y_source_train, x_target_train, y_target_train, x_lan_t,
                                y_target_dev, i + 1, source_domain)
                    # Test
                    AUROC, f1_value, target_acc = Test(x_target_test, y_target_test, i + 1)
                    AUROC_total = AUROC_total + AUROC
                    target_acc_total = target_acc + target_acc_total
                    f1_value_total = f1_value + f1_value_total
                sess.close()

            AUROC_av = AUROC_total / 10
            target_acc_av = target_acc_total / 10
            f1_value_av = f1_value_total / 10
            return AUROC_av, target_acc_av, f1_value_av

        AUROC_av, target_acc_av, f1_value_av = cross(x_target, y_target)
        print("AUROC_av", AUROC_av, "f1_value_av", f1_value_av, "target_acc_av", target_acc_av)
        file_object = open('./cnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(supervise) + '',
                           'a')

        strint = "result  :" + "AUC: " + str(
            AUROC_av) + "f1:" + str(f1_value_av) + "target_acc: " + str(target_acc_av) + "\n"

        file_object.write(strint)
        file_object.close()
        return AUROC_av, target_acc_av, f1_value_av


target_cell = sys.argv[1]
target_tf = sys.argv[2]
supervise = sys.argv[3]

AUROC_av, target_acc_av, f1_value_av = train_model(target_cell=target_cell, target_tf=target_tf, supervise=supervise)
