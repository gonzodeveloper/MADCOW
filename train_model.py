from utils import etl
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True,
                        choices=['variational', 'de-noising', 'conv', 'dense'],
                        help="type of model to train")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="location of model_params.json, will also be used to save model.h5 upon completion")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="location of saved OpenFAST simulation data")
    group_1 = parser.add_mutually_exclusive_group()
    group_1.add_argument("--single", action="store_true",
                         help="train on single turbine dataset")
    group_1.add_argument("--farm", action="store_true",
                         help="train on windfarm dataset")
    group_2 = parser.add_mutually_exclusive_group()
    group_2.add_argument("--build_cache", action="store_true",
                         help="build a cache in the data directory")
    group_2.add_argument("--load_cache", action="store_true",
                         help="load cache from data directory")

    return vars(parser.parse_args())


if __name__ == '__main__':
    '''
    Training will be fixed at 60 seconds windows 1 seconds sampling frequency!!
    '''
    args = parse_args()
    model_type = args['model_type']
    model_dir = args['model_dir']
    data_dir = args['data_dir']

    data_config = 'single' if args['single'] else 'farm'
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

    # Print Shape
    print("Training data shape  (n_windows, timesteps, features): ", scada_windows.shape)

    # Set validation data (will not change through training)
    training_windows, validation_windows = train_test_split(scada_windows, test_size=0.05)

    from models.ae import Seq_Autoencoder
    # Build Sequential auto encoder
    model = Seq_Autoencoder(log_dir=model_dir)
    model.build_model(ae_type=model_type,
                      timesteps=scada_windows.shape[1],
                      features=scada_windows.shape[2],
                      lstm_units=32,
                      hidden_units=16,
                      latent_dim=4,
                      dropout=0.1)

    # Fix validation data for training
    model.load_validation_data(validation_windows)
    model.load_training_data(training_windows)

    histories = model.train_model(num_epochs=25, epoch_id=0, batch_size=5000)
    model.save_model()

    df = pd.DataFrame(histories)
    df.to_csv(model_dir + "history.csv")