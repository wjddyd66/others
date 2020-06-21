# Initialize the data commonly used in the model in advance
import data_helper
import os
import pickle
import numpy as np
import sys
import gzip

def data_initialize(target_cell, target_tf):
    data_path = './data/Model/' + target_cell + '/' + target_tf + '/'
    x_source_file_path = data_path + 'x_source.pickle'
    y_source_file_path = data_path + 'y_source.pickle'
    x_target_file_path = data_path + 'x_target.pickle'
    y_target_file_path = data_path + 'y_target.pickle'

    if (os.path.isfile(x_source_file_path) and os.path.isfile(y_source_file_path)
            and os.path.isfile(x_target_file_path) and os.path.isfile(y_target_file_path)):

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

        return x_source, y_source, x_target, y_target

    else:
        print("NO Existing Data.....")

        # Load data
        print("Loading source and target data...")
        x_source, y_source, x_target, y_target = data_helper.load_data_and_labels(target_cell, target_tf)

        # Source Data
        print("Loading source one hot...")
        x_source = data_helper.load_one_hot(x_source)
        x_source, y_source = data_helper.shuffle_data(x_source, y_source)

        # Save Source Data
        with gzip.open(x_source_file_path, 'wb') as sx:
            print('Size of x_source: ', sys.getsizeof(x_source))
            pickle.dump(x_source, sx)
        with gzip.open(y_source_file_path, 'wb') as sy:
            pickle.dump(y_source, sy)

        # Target Data
        x_target, y_target = data_helper.shuffle_data(x_target, y_target)
        print("Loading target one hot...")
        x_target = data_helper.load_one_hot(x_target)
        x_target = np.array(x_target)

        # Save Target Data
        with gzip.open(x_target_file_path, 'wb') as tx:
            print('Size of x_target: ', sys.getsizeof(x_target))
            pickle.dump(x_target, tx)
        with gzip.open(y_target_file_path, 'wb') as ty:
            pickle.dump(y_target, ty)

        print("Saving Data set Finished....")

        return x_source, y_source, x_target, y_target


if __name__ == '__main__':
    Cell = ["K562", "H1-hESC", "HeLa-S3", "HepG2", "K562"]
    TF = ["CTCF", "GABP", "JunD", "REST", "USF2"]

    for c in Cell:
        for t in TF:
            target_cell = c
            target_tf = t
            x_source, y_source, x_target, y_target = data_initialize(target_cell, target_tf)

