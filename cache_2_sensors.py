from utils import etl
import numpy as np
import argparse


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="location of saved OpenFAST simulation data")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="location to save cache")
    args = vars(parser.parse_args())

    data_dir = args['data_dir']
    cache_dir = args['cache_dir']

    print("Finding Files")
    # First load all wind data into dataframes (sampling at 1 second frequency)
    openfast_outfiles = etl.get_output_files(data_dir)

    print("Loading simulations")
    scada_dfs = etl.multi_load(openfast_outfiles, frequency=1, config='single', columns=['Time', 'RotSpeed', 'GenPwr'])

    print("Transforming DFs")
    # Training data will be windows of 60 seconds in length (throw out any corrupt samples without 600 sec runtime)
    scada_windows = np.array([etl.create_windows(df, window_len=60)[:500] for df in scada_dfs if len(df) == 601])

    # Stack all windows and normalize by sensor
    scada_windows = np.vstack(scada_windows)

    etl.build_cache(data=scada_windows, data_dir=cache_dir)
