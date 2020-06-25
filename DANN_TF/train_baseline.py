import tensorflow as tf
import numpy as np
import os
import pickle
import pandas as pd
import data_helper
from model_baseline import DANN
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from utils import *
import sys
import data_initialization

def train_model_baseline(target_cell, target_tf, supervise):
    # Training parameters
    tf.flags.DEFINE_integer("dev_batch_size", 2048, "dev_Batch Size (default: 2048)")
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("evaluate_every", 10000, "Evaluate model on dev set after this many steps (default: 100)")
    # tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_integer("globel_flag", supervise, "Number of checkpoints to store (default: 5)")
    # Misc Parameters
    # log_device_placement=True
    # allow_soft_placement=True
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags()
    FLAGS(sys.argv)

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

#######################################################################################################################
    # Load data
    x_source, y_source, x_target, y_target = data_initialization.data_initialize(target_cell, target_tf)

    target_len = y_target.__len__()
    print("x_source", x_source.__len__())
    print("y_target", y_target.__len__())

    x_target, y_target = data_helper.shuffle_data(x_target, y_target)

    x_source_train = np.expand_dims(np.array(x_source), axis=-1)
    y_source_train = np.array(y_source)

    '''
    # Load data
    print("Loading source and target data...")
    x_source, y_source, x_target, y_target = data_helper.load_data_and_labels(target_cell, target_tf)
    source_len = y_source.__len__()
    target_len = y_target.__len__()
    print("x_source", x_source.__len__())
    print("y_target", y_target.__len__())
    print("Loading source one hot...")
    x_source = data_helper.load_one_hot(x_source)
    print("x_source", x_source.__len__())

    x_source, y_source = data_helper.shuffle_data(x_source, y_source)
    x_source_train = x_source
    y_source_train = y_source

    print("x_source_train shape", np.array(x_source_train).shape)
    print("y_source_train shape", np.array(y_source_train).shape)

    x_target, y_target = data_helper.shuffle_data(x_target, y_target)
    x_target_save = np.array(x_target)
    y_target_save = np.array(y_target)
    with open('./train/nocnn/' + target_tf + '_' + target_cell + '_' + str(supervise) + 'x.pkl', 'wb') as fx:
        pickle.dump(x_target_save, fx)
    with open('./train/nocnn/' + target_tf + '_' + target_cell + '_' + str(supervise) + 'y.pkl', 'wb') as fy:
        pickle.dump(y_target_save, fy)

    print("Loading target one hot...")
    x_target = data_helper.load_one_hot(x_target)
    x_target = np.array(x_target)
    print("x_target", x_target.__len__())
    x_source_train = np.expand_dims(np.array(x_source_train), axis=-1)
    y_source_train = np.array(y_source_train)
    '''


########################################################################################################################

    # Build the model graph
    checkpoint_dir = os.path.abspath(
        os.path.join(os.path.curdir, 'checkpoints/nocnn/' + target_tf + '_' + target_cell + '_' + str(supervise) + ''))
    # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True

        model = DANN(batch_size=FLAGS.batch_size)
        # model = DANN(batch_size=FLAGS.batch_size)

        learning_rate = 1e-3
        global_step = tf.Variable(0, name="global_step", trainable=False)
        pred_loss = tf.reduce_mean(model.pred_loss)
        regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
        # Evaluation
        correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
        label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
        # ROC AUC
        AUC_label = tf.argmax(model.classify_labels, 1)
        pred_label = tf.argmax(model.pred, 1)
        AUC_score = tf.split(model.pred, num_or_size_splits=2, axis=1)

        def cross(x_target, y_target):
            AUROC_total = 0.0
            target_acc_total = 0.0
            f1_value_total = 0.0
            for i in range(10):

                if (i != 9 & i != 0):
                    sess = tf.Session(config=session_conf)
                    with sess.as_default():

                        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it

                        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                        sess.run(tf.global_variables_initializer())

                        def train_step(x_batch, y_batch):

                            _, ploss, p_acc = \
                                sess.run([regular_train_op, pred_loss, label_acc],
                                         feed_dict={model.X: x_batch, model.y: y_batch,
                                                    })

                        def dev_step(x_batch_target, y_batch_target):
                            # """
                            # Evaluates model on a dev set
                            # """
                            acc, auc_label, auc_score = sess.run([label_acc, AUC_label, AUC_score],
                                                                 feed_dict={model.X: x_batch_target,
                                                                            model.y: y_batch_target,
                                                                            })
                            AUROC = roc_auc_score(auc_label, auc_score[1])
                            target_acc = acc
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
                                           })
                            AUROC = roc_auc_score(auc_label, auc_score[1])
                            data_label = pd.DataFrame(auc_label)
                            data_score = pd.DataFrame(auc_score[1])
                            data_label.to_csv('./result/auROC/nocnn/label/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            data_score.to_csv('./result/auROC/nocnn/score/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            f1 = f1_score(auc_label, f1_label)
                            target_acc = acc

                            file_object = open(
                                './result/nocnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(
                                    supervise) + '', 'a')

                            strint = "Test  :" + "AUC: " + str(
                                AUROC) + "f1 :" + str(f1) + "target_acc: " + str(target_acc) + "\n"

                            file_object.write(strint)
                            file_object.close()
                            return AUROC, f1, target_acc

                        def trainAndDev(p_x_source_train, p_y_source_train, p_x_target_train, p_y_target_train,
                                        p_x_lan_t,
                                        p_y_target_dev, step, flag):

                            best_target_AUROC = 0.6
                            checkpoint_prefix = os.path.join(checkpoint_dir + "/" + str(step), "model")
                            target_len = p_x_target_train.__len__()
                            x_train = []
                            y_train = []
                            if (flag == 1):
                                p_x_target_train = p_x_target_train[:int(target_len * (1 / 10))]
                                p_y_target_train = p_y_target_train[:int(target_len * (1 / 10))]

                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)

                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 2):
                                p_x_target_train = p_x_target_train[:int(target_len * (2 / 10))]
                                p_y_target_train = p_y_target_train[:int(target_len * (2 / 10))]
                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)

                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 5):
                                p_x_target_train = p_x_target_train[:int(target_len * (5 / 10))]
                                p_y_target_train = p_y_target_train[:int(target_len * (5 / 10))]
                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)

                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 10):
                                p_x_target_train = p_x_target_train
                                p_y_target_train = p_y_target_train
                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)
                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 0):
                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                x_train = p_x_source_train
                                y_train = p_y_source_train
                                print("x_train", x_train.shape)
                            for epoch in range(FLAGS.num_epochs):
                                #     # Generate batches
                                print('\nbegin trainAndDev')
                                gen_source_batch = data_helper.batch_iter(
                                    list(zip(x_train, y_train)), int(FLAGS.batch_size))

                                X0 = []
                                y0 = []

                                for batch in gen_source_batch:
                                    X, y = zip(*batch)
                                    X0.append(X)
                                    y0.append(y)
                                X0 = np.array(X0)

                                y0 = np.array(y0)

                                print("X0.shape", X0.shape)
                                print("y0.shape", y0.shape)

                                file_object = open(
                                    './result/nocnn/dev/devValue_' + target_tf + '_' + target_cell + '_' + str(
                                        supervise) + '', 'a')

                                for j in range(X0.shape[0]):
                                    x_source_i = X0[j]
                                    #  print("x_source_i .shape", x_source_i.shape)
                                    y_source_i = y0[j]
                                    x_r = x_source_i
                                    y_r = y_source_i
                                    train_step(x_r, y_r)
                                    if j % FLAGS.evaluate_every == 0:
                                        print("\nEvaluation")

                                        target_AUROC, target_acc = dev_step(p_x_lan_t, p_y_target_dev)

                                        if best_target_AUROC < target_AUROC:
                                            best_target_AUROC = target_AUROC
                                            path = saver.save(sess, checkpoint_prefix,
                                                              global_step=(epoch + 1) * (j + 1))
                                            print("Saved model checkpoint to {}\n".format(path))

                                        strint = "ecoch :" + str(epoch) + "  Evaluation  :" + str(
                                            j / FLAGS.evaluate_every) + "AUC: " + str(
                                            target_AUROC) + "target_acc: " + str(target_acc) + "\n"

                                        file_object.write(strint)

                                        print("")
                                file_object.close()

                            print("wwwwwwwww")

                        x_target_test = x_target[int(target_len * (i / 10)):int(target_len * ((i + 1) / 10))]
                        y_target_test = y_target[int(target_len * (i / 10)):int(target_len * ((i + 1) / 10))]
                        x_target_dev = x_target[int(target_len * ((i + 1) / 10)):int(target_len * ((i + 2) / 10))]
                        y_target_dev = y_target[int(target_len * ((i + 1) / 10)):int(target_len * ((i + 2) / 10))]
                        x_target_train = np.concatenate(
                            (x_target[:int(target_len * (i / 10))], x_target[int(target_len * ((i + 2) / 10)):]))
                        y_target_train = np.concatenate(
                            (y_target[:int(target_len * (i / 10))], y_target[int(target_len * ((i + 2) / 10)):]))
                        x_target_train = np.expand_dims(np.array(x_target_train), axis=-1)
                        y_target_train = np.array(y_target_train)

                        x_target_dev = np.expand_dims(np.array(x_target_dev), axis=-1)
                        y_target_dev = np.array(y_target_dev)

                        x_target_test = np.expand_dims(np.array(x_target_test), axis=-1)
                        y_target_test = np.array(y_target_test)
                        x_lan_t = x_target_dev
                        if not os.path.exists(checkpoint_dir + "/" + str(i + 1)):
                            os.makedirs(checkpoint_dir + "/" + str(i + 1))
                        trainAndDev(x_source_train, y_source_train, x_target_train, y_target_train, x_lan_t,
                                    y_target_dev, i + 1, FLAGS.globel_flag)
                        AUROC, f1_value, target_acc = Test(x_target_test, y_target_test, i + 1)
                        AUROC_total = AUROC_total + AUROC
                        target_acc_total = target_acc + target_acc_total
                        f1_value_total = f1_value + f1_value_total
                    sess.close()
                elif (i == 0):
                    sess = tf.Session(config=session_conf)
                    with sess.as_default():

                        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it

                        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                        sess.run(tf.global_variables_initializer())

                        def train_step(x_batch, y_batch):

                            _, ploss, p_acc = \
                                sess.run([regular_train_op, pred_loss, label_acc],
                                         feed_dict={model.X: x_batch, model.y: y_batch,
                                                    })

                        def dev_step(x_batch_target, y_batch_target):
                            # """
                            # Evaluates model on a dev set
                            # """
                            acc, auc_label, auc_score = sess.run([label_acc, AUC_label, AUC_score],
                                                                 feed_dict={model.X: x_batch_target,
                                                                            model.y: y_batch_target,
                                                                            })
                            AUROC = roc_auc_score(auc_label, auc_score[1])
                            target_acc = acc
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
                                           })
                            AUROC = roc_auc_score(auc_label, auc_score[1])
                            data_label = pd.DataFrame(auc_label)
                            data_score = pd.DataFrame(auc_score[1])
                            data_label.to_csv('./result/auROC/nocnn/label/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            data_score.to_csv('./result/auROC/nocnn/score/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            f1 = f1_score(auc_label, f1_label)
                            target_acc = acc

                            file_object = open(
                                './result/nocnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(
                                    supervise) + '', 'a')

                            strint = "Test  :" + "AUC: " + str(
                                AUROC) + "f1 :" + str(f1) + "target_acc: " + str(target_acc) + "\n"

                            file_object.write(strint)
                            file_object.close()
                            return AUROC, f1, target_acc

                        def trainAndDev(p_x_source_train, p_y_source_train, p_x_target_train, p_y_target_train,
                                        p_x_lan_t,
                                        p_y_target_dev, step, flag):

                            best_target_AUROC = 0.6
                            checkpoint_prefix = os.path.join(checkpoint_dir + "/" + str(step), "model")
                            target_len = p_x_target_train.__len__()
                            x_train = []
                            y_train = []
                            if (flag == 1):
                                p_x_target_train = p_x_target_train[:int(target_len * (1 / 10))]
                                p_y_target_train = p_y_target_train[:int(target_len * (1 / 10))]

                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)

                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                print("p_x_source_train", p_x_source_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 2):
                                p_x_target_train = p_x_target_train[:int(target_len * (2 / 10))]
                                p_y_target_train = p_y_target_train[:int(target_len * (2 / 10))]
                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)

                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 5):
                                p_x_target_train = p_x_target_train[:int(target_len * (5 / 10))]
                                p_y_target_train = p_y_target_train[:int(target_len * (5 / 10))]
                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)

                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 10):
                                p_x_target_train = p_x_target_train
                                p_y_target_train = p_y_target_train
                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)
                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 0):
                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                x_train = p_x_source_train
                                y_train = p_y_source_train
                                print("x_train", x_train.shape)
                            for epoch in range(FLAGS.num_epochs):
                                #     # Generate batches
                                print('\nbegin trainAndDev')
                                gen_source_batch = data_helper.batch_iter(
                                    list(zip(x_train, y_train)), int(FLAGS.batch_size))

                                X0 = []
                                y0 = []

                                for batch in gen_source_batch:
                                    X, y = zip(*batch)
                                    X0.append(X)
                                    y0.append(y)
                                X0 = np.array(X0)

                                y0 = np.array(y0)

                                print("X0.shape", X0.shape)
                                print("y0.shape", y0.shape)
                                file_object = open(
                                    './result/nocnn/dev/devValue_' + target_tf + '_' + target_cell + '_' + str(
                                        supervise) + '', 'a')

                                for j in range(X0.shape[0]):
                                    p = float(j) / X0.shape[0]
                                    l = 2. / (1. + np.exp(-10. * p)) - 1
                                    x_source_i = X0[j]
                                    #  print("x_source_i .shape", x_source_i.shape)
                                    y_source_i = y0[j]
                                    x_r = x_source_i
                                    y_r = y_source_i
                                    train_step(x_r, y_r)
                                    if j % FLAGS.evaluate_every == 0:
                                        print("\nEvaluation")

                                        target_AUROC, target_acc = dev_step(p_x_lan_t, p_y_target_dev)

                                        if best_target_AUROC < target_AUROC:
                                            best_target_AUROC = target_AUROC
                                            path = saver.save(sess, checkpoint_prefix,
                                                              global_step=(epoch + 1) * (j + 1))
                                            print("Saved model checkpoint to {}\n".format(path))

                                        strint = "ecoch :" + str(epoch) + "  Evaluation  :" + str(
                                            j / FLAGS.evaluate_every) + "AUC: " + str(
                                            target_AUROC) + "target_acc: " + str(target_acc) + "\n"

                                        file_object.write(strint)

                                        print("")
                                file_object.close()

                            print("wwwwwwwww")

                        x_target_test = x_target[int(target_len * 0):int(target_len * 0.1)]
                        y_target_test = y_target[int(target_len * 0):int(target_len * 0.1)]
                        x_target_dev = x_target[int(target_len * 0.1):int(target_len * 0.2)]
                        y_target_dev = y_target[int(target_len * 0.1):int(target_len * 0.2)]
                        x_target_train = x_target[int(target_len * 0.2):]
                        y_target_train = y_target[int(target_len * 0.2):]
                        x_target_train = np.expand_dims(np.array(x_target_train), axis=-1)

                        y_target_train = np.array(y_target_train)

                        x_target_dev = np.expand_dims(np.array(x_target_dev), axis=-1)
                        y_target_dev = np.array(y_target_dev)

                        x_target_test = np.expand_dims(np.array(x_target_test), axis=-1)
                        y_target_test = np.array(y_target_test)

                        x_lan_t = x_target_dev
                        if not os.path.exists(checkpoint_dir + "/" + str(i + 1)):
                            os.makedirs(checkpoint_dir + "/" + str(i + 1))
                        trainAndDev(x_source_train, y_source_train, x_target_train, y_target_train, x_lan_t,
                                    y_target_dev, i + 1, FLAGS.globel_flag)
                        AUROC, f1_value, target_acc = Test(x_target_test, y_target_test, i + 1)
                        AUROC_total = AUROC_total + AUROC
                        target_acc_total = target_acc + target_acc_total
                        f1_value_total = f1_value_total + f1_value
                    sess.close()
                else:
                    sess = tf.Session(config=session_conf)
                    with sess.as_default():

                        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it

                        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                        sess.run(tf.global_variables_initializer())

                        def train_step(x_batch, y_batch):

                            _, ploss, p_acc = \
                                sess.run([regular_train_op, pred_loss, label_acc],
                                         feed_dict={model.X: x_batch, model.y: y_batch,
                                                    })

                        def dev_step(x_batch_target, y_batch_target):
                            # """
                            # Evaluates model on a dev set
                            # """
                            acc, auc_label, auc_score = sess.run([label_acc, AUC_label, AUC_score],
                                                                 feed_dict={model.X: x_batch_target,
                                                                            model.y: y_batch_target,
                                                                            })
                            AUROC = roc_auc_score(auc_label, auc_score[1])
                            target_acc = acc
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
                                           })
                            AUROC = roc_auc_score(auc_label, auc_score[1])
                            data_label = pd.DataFrame(auc_label)
                            data_score = pd.DataFrame(auc_score[1])
                            data_label.to_csv('./result/auROC/nocnn/label/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            data_score.to_csv('./result/auROC/nocnn/score/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            f1 = f1_score(auc_label, f1_label)
                            target_acc = acc

                            file_object = open(
                                './result/nocnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(
                                    supervise) + '', 'a')

                            strint = "Test  :" + "AUC: " + str(
                                AUROC) + "f1 :" + str(f1) + "target_acc: " + str(target_acc) + "\n"

                            file_object.write(strint)
                            file_object.close()
                            return AUROC, f1, target_acc

                        def trainAndDev(p_x_source_train, p_y_source_train, p_x_target_train, p_y_target_train,
                                        p_x_lan_t,
                                        p_y_target_dev, step, flag):

                            best_target_AUROC = 0.6
                            checkpoint_prefix = os.path.join(checkpoint_dir + "/" + str(step), "model")
                            target_len = p_x_target_train.__len__()
                            x_train = []
                            y_train = []
                            if (flag == 1):
                                p_x_target_train = p_x_target_train[:int(target_len * (1 / 10))]
                                p_y_target_train = p_y_target_train[:int(target_len * (1 / 10))]

                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)

                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 2):
                                p_x_target_train = p_x_target_train[:int(target_len * (2 / 10))]
                                p_y_target_train = p_y_target_train[:int(target_len * (2 / 10))]
                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)

                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 5):
                                p_x_target_train = p_x_target_train[:int(target_len * (5 / 10))]
                                p_y_target_train = p_y_target_train[:int(target_len * (5 / 10))]
                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)

                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 10):
                                p_x_target_train = p_x_target_train
                                p_y_target_train = p_y_target_train
                                p_x_target_train = np.array(p_x_target_train)
                                p_y_target_train = np.array(p_y_target_train)
                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                print("p_x_target_train", p_x_target_train.shape)
                                print("p_y_target_train", p_y_target_train.shape)
                                x_train = np.concatenate((p_x_target_train, p_x_source_train), axis=0)
                                y_train = np.concatenate((p_y_target_train, p_y_source_train), axis=0)
                                print("x_train", x_train.shape)
                            elif (flag == 0):
                                print("flaggggggggggggg", flag)
                                print("target_len", target_len)
                                x_train = p_x_source_train
                                y_train = p_y_source_train
                                print("x_train", x_train.shape)

                            for epoch in range(FLAGS.num_epochs):
                                #     # Generate batches
                                print('\nbegin trainAndDev')
                                gen_source_batch = data_helper.batch_iter(
                                    list(zip(x_train, y_train)), int(FLAGS.batch_size))

                                X0 = []
                                y0 = []

                                for batch in gen_source_batch:
                                    X, y = zip(*batch)
                                    X0.append(X)
                                    y0.append(y)
                                X0 = np.array(X0)

                                y0 = np.array(y0)

                                print("X0.shape", X0.shape)
                                print("y0.shape", y0.shape)

                                file_object = open(
                                    './result/nocnn/dev/devValue_' + target_tf + '_' + target_cell + '_' + str(
                                        supervise) + '', 'a')

                                for j in range(X0.shape[0]):
                                    p = float(j) / X0.shape[0]
                                    l = 2. / (1. + np.exp(-10. * p)) - 1
                                    x_source_i = X0[j]
                                    #  print("x_source_i .shape", x_source_i.shape)
                                    y_source_i = y0[j]
                                    x_r = x_source_i
                                    y_r = y_source_i
                                    train_step(x_r, y_r)
                                    if j % FLAGS.evaluate_every == 0:
                                        print("\nEvaluation")

                                        target_AUROC, target_acc = dev_step(p_x_lan_t, p_y_target_dev)

                                        if best_target_AUROC < target_AUROC:
                                            best_target_AUROC = target_AUROC
                                            path = saver.save(sess, checkpoint_prefix,
                                                              global_step=(epoch + 1) * (j + 1))
                                            print("Saved model checkpoint to {}\n".format(path))

                                        strint = "ecoch :" + str(epoch) + "  Evaluation  :" + str(
                                            j / FLAGS.evaluate_every) + "AUC: " + str(
                                            target_AUROC) + "target_acc: " + str(target_acc) + "\n"

                                        file_object.write(strint)

                                        print("")
                                file_object.close()

                            print("wwwwwwwww")

                        x_target_test = x_target[int(target_len * 0.9):]
                        y_target_test = y_target[int(target_len * 0.9):]
                        x_target_dev = x_target[:int(target_len * 0.1)]
                        y_target_dev = y_target[:int(target_len * 0.1)]
                        x_target_train = x_target[int(target_len * 0.1):int(target_len * 0.9)]
                        y_target_train = y_target[int(target_len * 0.1):int(target_len * 0.9)]
                        x_target_train = np.expand_dims(np.array(x_target_train), axis=-1)
                        y_target_train = np.array(y_target_train)

                        x_target_dev = np.expand_dims(np.array(x_target_dev), axis=-1)
                        y_target_dev = np.array(y_target_dev)

                        x_target_test = np.expand_dims(np.array(x_target_test), axis=-1)
                        y_target_test = np.array(y_target_test)
                        x_lan_t = x_target_dev
                        if not os.path.exists(checkpoint_dir + "/" + str(i + 1)):
                            os.makedirs(checkpoint_dir + "/" + str(i + 1))
                        trainAndDev(x_source_train, y_source_train, x_target_train, y_target_train, x_lan_t,
                                    y_target_dev, i + 1, FLAGS.globel_flag)
                        AUROC, f1_value, target_acc = Test(x_target_test, y_target_test, i + 1)
                        AUROC_total = AUROC_total + AUROC
                        target_acc_total = target_acc + target_acc_total
                        f1_value_total = f1_value_total + f1_value
                    sess.close()

            AUROC_av = AUROC_total / 10

            target_acc_av = target_acc_total / 10
            f1_value_av = f1_value_total / 10
            return AUROC_av, target_acc_av, f1_value_av

        print("dasdasdasd")
        AUROC_av, target_acc_av, f1_value_av = cross(x_target, y_target)
        print("AUROC_av", AUROC_av, "f1_value_av", f1_value_av, "target_acc_av", target_acc_av)
        file_object = open('./result/nocnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(supervise) + '',
                           'a')

        strint = "result  :" + "AUC: " + str(
            AUROC_av) + "f1:" + str(f1_value_av) + "target_acc: " + str(target_acc_av) + "\n"

        file_object.write(strint)
        file_object.close()
        return AUROC_av, target_acc_av, f1_value_av

# Memory Limitation
MEMORY_LIMIT = 2048
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=MEMORY_LIMIT)])
    except RuntimeError as e:
        print(e)

target_cell = sys.argv[1]
target_tf = sys.argv[2]
supervise = sys.argv[3]
AUROC_av, target_acc_av, f1_value_av = train_model_baseline(target_cell=target_cell, target_tf=target_tf,
                                                            supervise=supervise)
