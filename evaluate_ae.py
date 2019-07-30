from utils import etl, visualize
from utils.anomaly_generation import sensor_failure, sensor_offset, sensor_drift
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import pandas as pd
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly_type", type=str, required=True,
                        help="type of anomaly to test with ['failure', 'offset' 'drift']")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="location of model .h5 file (also used to save eval data")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="location of saved OpenFAST simulation data")
    group_2 = parser.add_mutually_exclusive_group()
    group_2.add_argument("--build_cache", action="store_true",
                         help="build a cache in the data directory")
    group_2.add_argument("--load_cache", action="store_true",
                         help="load cache from data directory")
    return vars(parser.parse_args())


if __name__ == '__main__':

    # Parse arguments
    args = parse_args()
    anomaly_type = args['anomaly_type']
    model_dir = args['model_dir']
    data_dir = args['data_dir']
    cache = 'build' if args['build_cache'] else 'load'

    from keras.utils import normalize

    if cache == 'build':
        print("Loading data from OpenFAST runs in dirctory: ", data_dir)
        scada_windows = etl.load_openfast_outputs(data_dir, sampling_freq=1, window_len=60, window_step=1,
                                                  config=data_config)

        # Cache data
        etl.build_cache(data=scada_windows, data_dir=data_dir)

    elif cache == 'load':
        print("Loading Cache")
        scada_windows = etl.load_cache(data_dir)
        scada_windows = normalize(scada_windows, axis=0)

    else:
        scada_windows = None

    # Ten percent of windows to include anomalies
    outlier_fraction = 0.5
    normal_windows, anomalous_windows = train_test_split(scada_windows, test_size=outlier_fraction)
    print(scada_windows.shape)

    print("Generating Anomalies")
    sensor_count = anomalous_windows.shape[-1]
    n_anomalous_windows = anomalous_windows.shape[0]

    # On sensor failure for each window
    sensor_anomaly_idxs = np.random.randint(sensor_count, size=n_anomalous_windows)
    anomaly_labels = np.zeros(shape=(n_anomalous_windows, sensor_count))

    # Set flag for lab
    for idx, sensor_index in enumerate(sensor_anomaly_idxs):

        anomaly_labels[idx][sensor_index] = 1
        if anomaly_type == 'failure':
            anomalous_windows[idx] = sensor_failure(anomalous_windows[idx], sensor_idx=sensor_index,
                                                    begin_timestep=np.random.randint(50))

        elif anomaly_type == 'offset':
            anomalous_windows[idx] = sensor_offset(anomalous_windows[idx], sensor_idx=sensor_index,
                                                   begin_timestep=0, offset=np.random.uniform(1.10, 2.00))

        elif anomaly_type == "drift":
            anomalous_windows[idx] = sensor_drift(window=anomalous_windows[idx], sensor_idx=sensor_index,
                                                  begin_timestep=np.random.randint(0, 45), offset_step=0.05)

    print("Loading SEQ AE")
    from models.ae_model import Seq_Autoencoder
    seq_ae = Seq_Autoencoder(log_dir=model_dir, load_model=True)
    seq_ae.autoencoder.summary()

    normal_labels = np.zeros(shape=(normal_windows.shape[0], sensor_count))

    windows = np.vstack((normal_windows, anomalous_windows))
    labels = np.vstack((normal_labels, anomaly_labels))

    window_predictions = seq_ae.predict(windows)
    # We only care about reconstruction error on the final timestep
    real_frames = windows[::, -1, ::]
    pred_frames = window_predictions[::, -1, ::]

    reconstruction_errs = np.abs(pred_frames - real_frames)

    errs = reconstruction_errs.flatten()
    labels = labels.flatten()

    scores = []
    thresholds = np.linspace(0, 0.2, 100)
    for threshold in thresholds:
        round_err = [1 if err >= threshold else 0 for err in errs]
        scores.append(f1_score(labels, round_err))

    results_df = pd.DataFrame({'labels': labels, 'reconstruction_err': errs})
    results_df.to_csv(model_dir + "errors_and_labels.csv")
    # visualize.plot_reconstruction_errors(labels, errs, model_dir + "reconstruction_dist.png")
    visualize.plot_threshold_f1(thresholds, scores, model_dir + "threshold_plt.png")
