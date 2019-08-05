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
    parser.add_argument("--cache_type", type=str, required=True,
                        help="type of cache to build")
    args = vars(parser.parse_args())

    data_dir = args['data_dir']
    cache_dir = args['cache_dir']
    cache_type = args['cache_type']

    print("Finding Files")
    # First load all wind data into dataframes (sampling at 1 second frequency)
    openfast_outfiles = etl.get_output_files(data_dir)

    if cache_type == 'pair-wise':
        print("Loading simulations")
        scada_dfs = etl.multi_load(openfast_outfiles, frequency=1, config='single',
                                   columns=['Time', 'RotSpeed', 'GenPwr'])
        print("Transforming DFs")
        # Training data will be windows of 60 seconds in length (throw out any corrupt samples without 600 sec runtime)
        scada_windows = np.array([etl.create_windows(df, window_len=60)[:500] for df in scada_dfs if len(df) == 601])

        # Stack all windows and normalize by sensor
        scada_data = np.vstack(scada_windows)

    elif cache_type == 'step-wise':
        scada_dfs = etl.multi_load(openfast_outfiles, frequency=1, config='single')
        scada_data = [df.to_numpy() for df in scada_dfs]
        scada_data = np.vstack(scada_data)

    else:

        print("Loading simulations")
        scada_dfs = etl.multi_load(openfast_outfiles, frequency=1, config='single')
        print("Transforming DFs")
        # Training data will be windows of 60 seconds in length (throw out any corrupt samples without 600 sec runtime)
        scada_windows = np.array([etl.create_windows(df, window_len=60)[:500] for df in scada_dfs if len(df) == 601])

        # Stack all windows and normalize by sensor
        scada_data = np.vstack(scada_windows)

    print("Caching SCADA data. Size = ", scada_data.shape)
    etl.build_cache(data=scada_data, data_dir=cache_dir)
