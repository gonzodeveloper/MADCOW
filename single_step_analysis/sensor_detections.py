from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
import pickle


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


def sensor_fault(sensor_stack, sensor_idx, offset=0, failure=False):
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


def train_regressors(train_data):
    input_sensors = train_data[::, :5]

    rotspeed = train_data[::, 5]
    genspeed = train_data[::, 6]
    genpwr = train_data[::, 7]
    output_sensors = [rotspeed, genspeed, genpwr]

    regressors = [LogisticRegression(C=1.0, class_weight={0: 0.99, 1: 0.01}, solver='saga', n_jobs=-1)
                  .fit(input_sensors, out_sensor) for out_sensor in output_sensors]
    return regressors


def similarity(x, y):
    return np.abs(x - y)

def predict_at_threshold(test_data, regressors, threshold):
    input_sensors = test_data[::, :5]

    rotspeed = test_data[::, 5]
    genspeed = test_data[::, 6]
    genpwr = test_data[::, 7]
    truth = np.concatenate([rotspeed, genspeed, genpwr])

    rotspeed_preds = regressors[0].predict(input_sensors)
    genspeed_preds = regressors[1].predict(input_sensors)
    genpwr_preds = regressors[2].predict(input_sensors)

    rotspeed_preds = [1 if similarity(speed, pred) > threshold else 0
                      for speed, pred in zip(rotspeed, rotspeed_preds)]
    genspeed_preds = [1 if similarity(speed, pred) > threshold else 0
                      for speed, pred in zip(genspeed, genspeed_preds)]
    genpwr_preds = [1 if similarity(pwr, pred) > threshold else 0
                      for pwr, pred in zip(genpwr, genpwr_preds)]

    preds = np.concatenate([rotspeed_preds, genspeed_preds, genpwr_preds])





def maximize_threshold(test_data, regressors):


    thresholds = np.linspace(0.01, 0.5, 100)

    scores = [predict_at_threshold(test_data, regressors, cutoff) for cutoff in thresholds]

if __name__ == '__main__':
    # Parse args
    args = parse_args()
    log_dir = args['log_dir']
    cache_dir = args['cache_dir']

    print("Loading data cache...")
    # Load data (first 1 million sensor stacks)
    scada_data = load_cache(cache_dir)[:50000]

    train_dat, test_dat = train_test_split(scada_data, test_size=0.2)

    print("Transforming and generating anomalies...")
    # Add anomalies to data
    normal, anoms = train_test_split(test_dat, test_size=0.01)

    anom_idxs = np.random.randint(5, 8, size=len(anoms))

    # Anomalies only in RotSpeed, GenSpeed, GenPwr
    anoms = np.array([sensor_fault(stack, sensor_idx=idx, failure=True)
                      for idx, stack in zip(anoms, anom_idxs)])
    scada_data = np.vstack((normal, anoms))

    # Split train test
    train_dat, test_dat = train_test_split(scada_data, test_size=0.1)

    print("Beginning parallel model training...")