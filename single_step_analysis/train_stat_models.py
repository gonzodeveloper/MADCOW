from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from pyod.models.cblof import CBLOF
from pyod.models.auto_encoder import AutoEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import pickle
import argparse
import sys


def parse_args():
    """
    Parse command line arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="directory containing dataset cache")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="directory to log results")
    parser.add_argument("--failure", action='store_true',
                       help="if set this will generate sensor failures")
    parser.add_argument("--offset", type=float,
                       help="if set this specifies the decimal percent offset of the sensors")
    return vars(parser.parse_args())


def load_cache(data_dir):
    """
    Loads dataset from np.memmap on disk

    :param data_dir: absolute path to saved dataset
    :return: np.array containing data (n_windows, timesteps, features)
    """
    # Load shape from file
    with open(data_dir + "shape.pkl", "rb") as f:
        data_shape = pickle.load(f)

    fp = np.memmap(data_dir + "cache.dat", dtype='float32', mode='r', shape=data_shape)
    return fp


def sensor_fault(sensor_stack, sensor_idx, offset=0.0, failure=False):
    """
    Creates an anomoly in the sensor stack, at a specified index create a sensor failure or offset

    :param sensor_stack: np.array (1D) stack of sensor data
    :param sensor_idx: index of sensor to create anomaly
    :param offset: decimal offset to apply to sensor (e.g. 0.5 gives a 50% increase to sensor value)
    :param failure: if true will set the given sensor to zero (cannot be used with offset)
    :return: np.array anomalous sensor stack
    """
    anomaly = np.array(sensor_stack)
    if failure is True:
        anomaly[sensor_idx] = 0
    else:
        anomaly[sensor_idx] *= (1 + offset)
    return anomaly


def train_and_evaluate(model, model_name, X_train, X_test, y_test, contamination):
    """
    Trains a model, evaluates with testing data and writes report
    :param model: pyod outlier detection model (e.g. KNN, SVM, Autoencoder, etc.)
    :param model_name: name of model for report; string
    :param X_train: training data; np.arrays
    :param X_test: testing data; np.arrays
    :param y_test: ground truth for testing data (0s for inliers 1s for outliers)
    :param decimal fraction of outlier contamination (approx)
    :return: report containing model name,
    """
    print("Training Model: ",  model_name)
    sys.stdout.flush()

    model.fit(X_train)

    print("Getting metrics for model: ", model_name)
    sys.stdout.flush()

    y_test_pred = model.predict(X_test)
    # y_test_pred = model.decision_function(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auroc = roc_auc_score(y_test, y_test_pred)

    report = {'1_model_name': model_name, 'accuracy_score': acc, 'recall_score': recall, 'precision_score': precision,
              'f1_score': f1, 'AUROC': auroc, 'contamination': contamination}
    print(report)
    return report


def generate_train_test(data, contamination, sensor_failure=True, offset_pct=None, test_pct=0.2):
    """
    Generate labeled train and test sets for data with a given contamination percentage.
    Anomalies are generates as either sensor failures or offsets of rotation speed, generator speed,
    or generator power

    :param data: 2D np.array of sensor data (n_frames x sensors) where we have 8 sensors.
                wind velocity x, y, and z, blade pitch, nac yaw, rotation speed, generator speed,
                and generator power
    :param contamination: decimal fraction of data to transform into anomalous
    :param sensor_failure: boolean val, if true randomly generate sensor failures
    :param offset_pct: if not failure, define a decimal offset value
    :param test_pct: percent of data to leave aside for testing
    :return: np.arrays x_train, x_test, y_train, y_test
    """
    # Split by contamination fraction (training includes anomalies)
    norms, anoms = train_test_split(data, test_size=contamination)
    # Anomalies only in RotSpeed, GenSpeed, GenPwr
    anoms = np.array([sensor_fault(stack, sensor_idx=np.random.randint(5, 8),
                                   failure=sensor_failure, offset=offset_pct)
                      for stack in anoms])
    # Generate labels
    normal_labels = np.zeros(len(norms))
    anom_labels = np.ones(len(anoms))
    labels = np.concatenate((normal_labels, anom_labels))

    # Stitch back together and split with test percentage
    data = np.vstack((norms, anoms))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_pct)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Parse args
    args = parse_args()
    log_dir = args['log_dir']
    cache_dir = args['cache_dir']
    fail = args['failure']
    offset = args['offset']

    print("Loading data cache...")
    # Load data (first 1 million sensor stacks)
    scada_data = load_cache(cache_dir)[:100000]
    scalar = MinMaxScaler(feature_range=(-1, 1))
    scalar.fit(scada_data)
    scada_data = scalar.transform(scada_data)

    contamination_fracs = [0.5, 0.4, 0.3, 0.2, 0.1, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01,
                           0.008, 0.005, 0.004, 0.003, 0.002, 0.001]
    all_report_dfs = []
    for anomaly_frac in contamination_fracs:
        print("Running models with {} contamination rate".format(anomaly_frac))
        # Instantiate models
        model_names = ['knn', 'pca', 'cblof', 'iforest', 'autoencoder']
        models = [
            KNN(contamination=anomaly_frac, algorithm='kd_tree', n_neighbors=13, n_jobs=8),
            PCA(contamination=anomaly_frac, svd_solver='auto', standardization=False),
            CBLOF(contamination=anomaly_frac, n_clusters=16, n_jobs=8),
            IForest(contamination=0.1, n_estimators=100, n_jobs=8, behaviour='new'),
            AutoEncoder(contamination=anomaly_frac, hidden_neurons=[4, 2, 2, 4], hidden_activation='tanh',
                        batch_size=5000, epochs=200, preprocessing=False, verbose=0)
        ]
        X_train, X_test, y_train, y_test = generate_train_test(scada_data, contamination=anomaly_frac,
                                                               sensor_failure=fail, offset_pct=offset)
        reports = [train_and_evaluate(model, name, X_train, X_test, y_test, anomaly_frac)
                   for model, name in zip(models, model_names)]
        reports_df = pd.DataFrame(reports)
        all_report_dfs.append(reports_df)

    fulll_report_df = pd.concat(all_report_dfs)
    print(fulll_report_df)
    fulll_report_df.to_csv(log_dir + "full_report.csv")



