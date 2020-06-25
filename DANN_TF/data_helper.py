import numpy as np
import re
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

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
    # print(negative_text_x.__len__())

    positive_text_x.extend(negative_text_x)
    # print(positive_text_x.__len__())

    x_text = [clean_data(s) for s in positive_text_x]
    y_labels = np.concatenate((positive_labels, negative_labels), axis=0)

    return x_text, y_labels


def load_data_and_labels(target_cell, target_tf):
    print(target_cell)
    source_cell = ["GM12878", "K562", "HeLa-S3", "HepG2", "H1-hESC"]
    tf_names = ["CTCF", "GABP", "JunD", "REST", "USF2"]
    source_cell.remove(target_cell)

    x_CTCF, y_CTCF = load_file("./data/" + source_cell[0] + "/" + target_tf + ".fasta",
                               "./data/" + source_cell[0] + "/" + target_tf + "_back.fasta")
    x_GABP, y_GABP = load_file("./data/" + source_cell[1] + "/" + target_tf + ".fasta",
                               "./data/" + source_cell[1] + "/" + target_tf + "_back.fasta")
    x_JunD, y_JunD = load_file("./data/" + source_cell[2] + "/" + target_tf + ".fasta",
                               "./data/" + source_cell[2] + "/" + target_tf + "_back.fasta")
    x_REST, y_REST = load_file("./data/" + source_cell[3] + "/" + target_tf + ".fasta",
                               "./data/" + source_cell[3] + "/" + target_tf + "_back.fasta")
    # target data
    x_USF2, y_USF2 = load_file("./data/" + target_cell + "/" + target_tf + ".fasta",
                               "./data/" + target_cell + "/" + target_tf + "_back.fasta")

    x_CTCF.extend(x_GABP)
    x_CTCF.extend(x_JunD)
    x_CTCF.extend(x_REST)
    x_source = x_CTCF
    print(x_source.__len__())
    y_source = np.concatenate((y_CTCF, y_GABP), axis=0)
    y_source = np.concatenate((y_source, y_JunD), axis=0)
    y_source = np.concatenate((y_source, y_REST), axis=0)
    print(y_source.__len__())
    x_target = x_USF2
    y_target = y_USF2
    # return x_REST, y_REST, x_target, y_target

    return x_source, y_source, x_target, y_target


def load_one_hot(data):
    # x_source,y_source,x_target,y_target=load_data_and_labels()
    tx = []
    enc = OneHotEncoder()
    enc.fit([[0] * 101, [1] * 101, [2] * 101, [3] * 101])
    for i in tqdm(range(data.__len__())):
        a = enc.transform([np.array(list(data[i]), dtype=int)]).toarray().reshape(101, 4)
        a = a.T
        tx.append(a)
    print(tx.__len__())

    return tx


def batch_iter(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    data_size = int(data_size / batch_size) * batch_size
    print("data size ", data_size)
    num_batch_per_epochs = int((data_size - 1) / batch_size) + 1
    print("num_batch_per_epochs", num_batch_per_epochs)
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


def shuffle_data(x_data, y_data):
    print("Loading data...")
    np.random.seed(100)
    shuffle_indices = np.random.permutation(np.arange(len(y_data)))
    x_s = []
    y_s = []
    for i in range(len(y_data)):
        x_s.append(x_data[shuffle_indices[i]])
        y_s.append(y_data[shuffle_indices[i]])
    return x_s, y_s