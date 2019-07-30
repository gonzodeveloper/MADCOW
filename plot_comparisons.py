from utils import etl, visualize
from utils.anomaly_generation import sensor_failure, sensor_offset, sensor_drift
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
import os

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly_type", type=str, required=True,
                        help="type of anomaly to test with ['failure', 'offset' 'drift']")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="location of model .h5 file (also used to save eval data")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="location of saved OpenFAST simulation data")
    args = vars(parser.parse_args())

    anomaly_type = args['anomaly_type']
    model_dir = args['model_dir']
    data_dir = args['cache_dir']

    # Load cache (ONLY USE 100)
    scada_windows = etl.load_cache(data_dir)[:100]

    # 50/50 Anomalies
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

    from models.ae import Seq_Autoencoder

    print("Loading SEQ AE")
    seq_ae = Seq_Autoencoder(log_dir=model_dir, load_model=True)
    seq_ae.autoencoder.summary()

    normal_preds = seq_ae.predict(normal_windows)
    anom_preds = seq_ae.predict(anomalous_windows)
    subtractions = np.abs(anomalous_windows - anom_preds)

    img_dir = model_dir + "plots/"
    try:
        os.mkdir(img_dir)
    except FileExistsError:
        pass

    for idx, (win, pred) in enumerate(zip(normal_windows, normal_preds)):
        visualize.plot_sensors("Normal", "Prediction", win, pred, img_dir + "normal_plot_{0:02d}".format(idx))

    for idx, (win, pred, sub) in enumerate(zip(anomalous_windows, anom_preds, subtractions)):
        visualize.plot_sensors("Anomaly", "Prediction", win, pred, img_dir + "anom_plot{0:02d}".format(idx))
        visualize.plot_sensors("Anomaly", "Reconstruction Error", win, sub, img_dir + "rcstr_plot_{0:02d}".format(idx))


