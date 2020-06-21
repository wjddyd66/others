import tensorflow as tf
import numpy as np
import os
import pickle
import pandas as pd
import data_helper
from model_DANN import DANN
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from utils import *
import random
import sys
import data_initialization

def train_model(target_cell, target_tf, supervise):
    sup_flag = 32
    if supervise == 0:
        sup_flag = 32
    elif supervise == 1:
        sup_flag == 35
    elif supervise == 2:
        sup_flag == 38
    elif supervise == 5:
        sup_flag == 48
    else:
        sup_flag = 64
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
    flags.DEFINE_integer("domain_classes", 2, "The number of categories for the domain (default: 2)")
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
    x_source, y_source, x_target, y_target = data_initialization.data_initialize(target_cell,target_tf)

    target_len = y_target.__len__()
    print("x_source", x_source.__len__())
    print("y_target", y_target.__len__())

    x_target, y_target = data_helper.shuffle_data(x_target, y_target)

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

                if (i != 9 & i != 0):
                    sess = tf.Session(config=session_conf)
                    with sess.as_default():
                        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
                        sess.run(tf.global_variables_initializer())

                        def train_step(x_batch, y_batch, domain_labels, l):

                            _, batch_loss, dloss, ploss, d_acc, p_acc = \
                                sess.run([dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
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
                            data_label.to_csv('./result/auROC/cnn/label/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            data_score.to_csv('./result/auROC/cnn/score/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            f1 = f1_score(auc_label, f1_label)
                            target_acc = acc

                            file_object = open(
                                './result/cnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(
                                    supervise) + '', 'a')

                            strint = "Test  :" + "AUC: " + str(
                                AUROC) + "f1 :" + str(f1) + "target_acc: " + str(target_acc) + "\n"

                            file_object.write(strint)
                            file_object.close()
                            return AUROC, f1, target_acc

                        def trainAndDev(p_x_source_train, p_y_source_train, p_x_target_train, p_y_target_train,
                                        p_x_lan_t,
                                        p_y_target_dev, step):

                            best_target_AUROC = 0.5
                            checkpoint_prefix = os.path.join(checkpoint_dir + "/" + str(step), "model")
                            for epoch in range(FLAGS.num_epochs):
                                #     # Generate batches
                                print('\nbegin trainAndDev')
                                gen_source_batch = data_helper.batch_iter(
                                    list(zip(p_x_source_train, p_y_source_train)), int(FLAGS.batch_size / 2))
                                gen_target_batch = data_helper.batch_iter(
                                    list(zip(p_x_target_train, p_y_target_train)), int(FLAGS.batch_size / 2))
                                domain_labels = np.vstack([np.tile([1, 0], [32, 1]),
                                                           np.tile([0, 1], [32, 1])])
                                print("domain_labels.shape", domain_labels.shape)
                                X0 = []
                                y0 = []
                                X1 = []
                                y1 = []
                                for batch in gen_source_batch:
                                    X, y = zip(*batch)
                                    X0.append(X)
                                    y0.append(y)
                                for batch in gen_target_batch:
                                    X, y = zip(*batch)
                                    X1.append(X)
                                    y1.append(y)
                                X0 = np.array(X0)
                                X1 = np.array(X1)
                                y0 = np.array(y0)
                                y1 = np.array(y1)
                                print("X0.shape", X0.shape)
                                print("y0.shape", y0.shape)
                                print("y1.shape", y1.shape)

                                x0_size = X0.shape[0]
                                x1_size = X1.shape[0]
                                x_i = []
                                y_i = []


                                for i in range(x0_size):
                                    x_temp = random.choice(range(x1_size))
                                    x = np.concatenate((X0[i], X1[x_temp]), axis=0)
                                    y_temp = np.concatenate((y0[i], y1[x_temp]), axis=0)
                                    x_i.append(x)
                                    y_i.append(y_temp)

                                x_i = np.array(x_i)
                                y_i = np.array(y_i)

                                print("x_i .shape", x_i.shape)
                                print("y_i .shape", y_i.shape)
                                file_object = open(
                                    './result/cnn/dev/devValue_' + target_tf + '_' + target_cell + '_' + str(
                                        supervise) + '', 'a')

                                for j in range(x_i.shape[0]):
                                    p = float(j) / x_i.shape[0]
                                    l = 2. / (1. + np.exp(-10. * p)) - 1
                                    # l = 1e-3
                                    x_source_i = x_i[j]
                                    #  print("x_source_i .shape", x_source_i.shape)
                                    y_source_i = y_i[j]
                                    x_r = x_source_i
                                    y_r = y_source_i
                                    train_step(x_r, y_r, domain_labels, l)
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

                        x_target_test = x_target[int(target_len * (i / 10)):int(target_len * ((i + 1) / 10))]
                        y_target_test = y_target[int(target_len * (i / 10)):int(target_len * ((i + 1) / 10))]
                        x_target_dev = x_target[int(target_len * ((i + 1) / 10)):int(target_len * ((i + 2) / 10))]
                        y_target_dev = y_target[int(target_len * ((i + 1) / 10)):int(target_len * ((i + 2) / 10))]
                        x_target_train = np.concatenate(
                            (x_target[:int(target_len * (i / 10))], x_target[int(target_len * ((i + 2) / 10)):]))
                        y_target_train = np.concatenate(
                            (y_target[:int(target_len * (i / 10))], y_target[int(target_len * ((i + 2) / 10)):]))
                        # x_target_train = x_target[:int(target_len * (i / 10))] + x_target[int(target_len * ((i + 2) / 10)):]
                        # y_target_train = y_target[:int(target_len * (i / 10))] + y_target[int(target_len * ((i + 2) / 10)):]
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
                                    y_target_dev, i + 1)
                        AUROC, f1_value, target_acc = Test(x_target_test, y_target_test, i + 1)
                        AUROC_total = AUROC_total + AUROC
                        target_acc_total = target_acc + target_acc_total
                        f1_value_total = f1_value + f1_value_total
                    sess.close()
                elif (i == 0):
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
                            data_label.to_csv('./result/auROC/cnn/label/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            data_score.to_csv('./result/auROC/cnn/score/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            target_acc = acc
                            f1 = f1_score(auc_label, f1_label)
                            file_object = open(
                                './result/cnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(
                                    supervise) + '', 'a')

                            strint = "Test  :" + "AUC: " + str(
                                AUROC) + "f1:" + str(f1) + "target_acc: " + str(target_acc) + "\n"

                            file_object.write(strint)
                            file_object.close()
                            return AUROC, f1, target_acc

                        def trainAndDev(p_x_source_train, p_y_source_train, p_x_target_train, p_y_target_train,
                                        p_x_lan_t,
                                        p_y_target_dev, step):

                            best_target_AUROC = 0.5
                            checkpoint_prefix = os.path.join(checkpoint_dir + "/" + str(step), "model")
                            for epoch in range(FLAGS.num_epochs):
                                #     # Generate batches
                                print('\nbegin trainAndDev')
                                gen_source_batch = data_helper.batch_iter(
                                    list(zip(p_x_source_train, p_y_source_train)), int(FLAGS.batch_size / 2))
                                gen_target_batch = data_helper.batch_iter(
                                    list(zip(p_x_target_train, p_y_target_train)), int(FLAGS.batch_size / 2))
                                domain_labels = np.vstack([np.tile([1, 0], [32, 1]),
                                                           np.tile([0, 1], [32, 1])])
                                print("domain_labels.shape", domain_labels.shape)

                                # Training step
                                # if training_mode == 'dann':
                                X0 = []
                                y0 = []
                                X1 = []
                                y1 = []
                                for batch in gen_source_batch:
                                    X, y = zip(*batch)

                                    X0.append(X)
                                    y0.append(y)
                                for batch in gen_target_batch:
                                    X, y = zip(*batch)
                                    X1.append(X)
                                    y1.append(y)
                                X0 = np.array(X0)
                                X1 = np.array(X1)
                                y0 = np.array(y0)
                                y1 = np.array(y1)
                                print("X0.shape", X0.shape)
                                print("y0.shape", y0.shape)
                                print("y1.shape", y1.shape)
                                x0_size = X0.shape[0]
                                x1_size = X1.shape[0]
                                x_i = []
                                y_i = []
                                for i in range(x0_size):
                                    x_temp = random.choice(range(x1_size))
                                    x = np.concatenate((X0[i], X1[x_temp]), axis=0)
                                    y_temp = np.concatenate((y0[i], y1[x_temp]), axis=0)
                                    x_i.append(x)
                                    y_i.append(y_temp)

                                x_i = np.array(x_i)
                                y_i = np.array(y_i)

                                print("x_i .shape", x_i.shape)
                                print("y_i .shape", y_i.shape)
                                file_object = open(
                                    './result/cnn/dev/devValue_' + target_tf + '_' + target_cell + '_' + str(
                                        supervise) + '', 'a')

                                for j in range(x_i.shape[0]):
                                    p = float(j) / x_i.shape[0]
                                    l = 2. / (1. + np.exp(-10. * p)) - 1
                                    # l = 1e-3
                                    x_source_i = x_i[j]
                                    # print("x_source_i .shape", x_source_i.shape)
                                    y_source_i = y_i[j]
                                    x_r = x_source_i
                                    y_r = y_source_i
                                    train_step(x_r, y_r, domain_labels, l)
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
                            # AUROC, target_acc=Test(p_x_target_test,p_y_target_test,step,sess)

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
                                    y_target_dev, i + 1)
                        AUROC, f1_value, target_acc = Test(x_target_test, y_target_test, i + 1)
                        AUROC_total = AUROC_total + AUROC
                        target_acc_total = target_acc + target_acc_total
                        f1_value_total = f1_value_total + f1_value
                    sess.close()
                else:
                    sess = tf.Session(config=session_conf)
                    with sess.as_default():
                        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                        sess.run(tf.global_variables_initializer())

                        def train_step(x_batch, y_batch, domain_labels, l):

                            # T=sess.run(True)
                            _, batch_loss, dloss, ploss, d_acc, p_acc = \
                                sess.run([dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
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
                            data_label.to_csv('./result/auROC/cnn/label/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            data_score.to_csv('./result/auROC/cnn/score/' + target_tf + '_' + target_cell + '_' + str(
                                supervise) + '.csv', mode='a', header=False, index=False)
                            target_acc = acc
                            f1 = f1_score(auc_label, f1_label)
                            file_object = open(
                                './result/cnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(
                                    supervise) + '', 'a')

                            strint = "Test  :" + "AUC: " + str(
                                AUROC) + "f1:" + str(f1) + "target_acc: " + str(target_acc) + "\n"

                            file_object.write(strint)
                            file_object.close()
                            return AUROC, f1, target_acc

                        def trainAndDev(p_x_source_train, p_y_source_train, p_x_target_train, p_y_target_train,
                                        p_x_lan_t,
                                        p_y_target_dev, step):

                            best_target_AUROC = 0.5
                            checkpoint_prefix = os.path.join(checkpoint_dir + "/" + str(step), "model")
                            for epoch in range(FLAGS.num_epochs):
                                #     # Generate batches
                                print('\nbegin trainAndDev')
                                gen_source_batch = data_helper.batch_iter(
                                    list(zip(p_x_source_train, p_y_source_train)), int(FLAGS.batch_size / 2))
                                gen_target_batch = data_helper.batch_iter(
                                    list(zip(p_x_target_train, p_y_target_train)), int(FLAGS.batch_size / 2))
                                domain_labels = np.vstack([np.tile([1, 0], [32, 1]),
                                                           np.tile([0, 1], [32, 1])])
                                print("domain_labels.shape", domain_labels.shape)

                                # Training step
                                X0 = []
                                y0 = []
                                X1 = []
                                y1 = []
                                for batch in gen_source_batch:
                                    X, y = zip(*batch)

                                    X0.append(X)
                                    y0.append(y)
                                for batch in gen_target_batch:
                                    X, y = zip(*batch)
                                    X1.append(X)
                                    y1.append(y)
                                X0 = np.array(X0)
                                X1 = np.array(X1)
                                y0 = np.array(y0)
                                y1 = np.array(y1)
                                print("X0.shape", X0.shape)
                                print("y0.shape", y0.shape)
                                print("y1.shape", y1.shape)
                                x0_size = X0.shape[0]
                                x1_size = X1.shape[0]
                                x_i = []
                                y_i = []
                                for i in range(x0_size):
                                    x_temp = random.choice(range(x1_size))
                                    x = np.concatenate((X0[i], X1[x_temp]), axis=0)
                                    y_temp = np.concatenate((y0[i], y1[x_temp]), axis=0)
                                    x_i.append(x)
                                    y_i.append(y_temp)

                                x_i = np.array(x_i)
                                y_i = np.array(y_i)

                                print("x_i .shape", x_i.shape)
                                print("y_i .shape", y_i.shape)
                                file_object = open(
                                    './result/cnn/dev/devValue_' + target_tf + '_' + target_cell + '_' + str(
                                        supervise) + '', 'a')

                                for j in range(x_i.shape[0]):
                                    p = float(j) / x_i.shape[0]
                                    l = 2. / (1. + np.exp(-10. * p)) - 1
                                    # l = 1e-3
                                    x_source_i = x_i[j]
                                    # print("x_source_i .shape", x_source_i.shape)
                                    y_source_i = y_i[j]
                                    x_r = x_source_i
                                    y_r = y_source_i
                                    train_step(x_r, y_r, domain_labels, l)
                                    if j % FLAGS.evaluate_every == 0:
                                        print("\nEvaluation")

                                        target_AUROC, target_acc = dev_step(p_x_lan_t, p_y_target_dev)

                                        if best_target_AUROC < target_AUROC:
                                            best_target_AUROC = target_AUROC
                                            path = saver.save(sess, checkpoint_prefix,
                                                              global_step=(epoch + 1) * (j + 1))
                                            print("Saved model checkpoint to {}\n".format(path))

                                        strint = "ecoch :" + str(epoch) + "Evaluation  :" + str(
                                            j / FLAGS.evaluate_every) + "AUC: " + str(
                                            target_AUROC) + "target_acc: " + str(target_acc) + "\n"

                                        file_object.write(strint)

                                        print("")
                                file_object.close()
                            # AUROC, target_acc=Test(p_x_target_test,p_y_target_test,step,sess)

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
                                    y_target_dev, i + 1)
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
        file_object = open('./result/cnn/test/TestValue_' + target_tf + '_' + target_cell + '_' + str(supervise) + '',
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
