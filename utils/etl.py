from multiprocessing import Pool
from floris.simulation import floris
import pandas as pd
import numpy as np
import pickle
import shutil
import re
import os
import glob
import json
import sys
import subprocess
import io


def get_output_files(data_dir):
    """
    Get all OpenFAST output files under parent's directory tree

    :param data_dir: absolute path to parent data directory
    :return: list of absolute paths to OpenFAST output files
    """
    return glob.glob(data_dir + "**/*.out")


def create_windows(df, window_len, step):
    """
    Transform a dataframe of OpenFAST output data into an array representing a sliding window of given length

    :param df: pandas df of OpenFAST data
    :param window_len: length of sliding window
    :param step: step increment for sliding windos
    :return: np.array of sliding windows
    """
    arr = df.to_numpy()
    return np.array([arr[i:i + window_len] for i in range(0, len(arr) - window_len - 1, step)])


def _load_single_openfast(filepath, frequency, columns=None):
    """
    Reads a single OpenFAST output file into a pandas dataframe.
    Samples are taken from the OpenFAST file at a given frequency.
    Dataframe contains 8 primary sensor values we are using.

    :param filepath: absolute filepath to OpenFAST output
    :param frequency: frequency in seconds of desired sampling
    :param columns: which sensor values to load
    :return: pandas dataframe of sensor outputs
    """

    if columns is None:
        columns = ['Time', 'Wind1VelX', 'Wind1VelY', 'Wind1VelZ', 'BldPitch1',
                           'NacYaw', 'RotSpeed', 'GenSpeed', 'GenPwr']
    try:
        df = pd.read_csv(filepath, sep='\t', header=3, low_memory=False)
        sim_df = df.drop(0)
        sim_df.columns = [col.strip() for col in sim_df.columns]

        scada_df = sim_df[columns]
        scada_df = scada_df.apply(pd.to_numeric)

        freq_mask = scada_df['Time'] % frequency == 0

        scada_df = scada_df[freq_mask]
        scada_df.set_index('Time', inplace=True)
        return scada_df

    except FileNotFoundError:
        return None


def load_simulation(filepaths, frequency, config, columns=None):
    """
    Loads a simulation's worth of OpenFAST data into a pandas dataframe.
    One filepath will return a single turbine simulation.
    Multiple filepaths will be merged into a single dataframe representing a farm of turbines.
    Samples will be taken from OpenFAST output at given frequency

    :param filepaths: absolure filepath(s) to OpenFAST output
    :param frequency: frequency in seconds of desired sampling
    :param config: 'single' or 'farm' indicates single turbine simulation of entire farm
    :param columns: which sensor values to load
    :return: pandas dataframe of sensor outputs for turbine(s)
    """
    if config == 'single':
        # Simply load single turbine's openfast output into dataframe
        return _load_single_openfast(filepaths, frequency, columns)
    elif config == 'farm':
        # Load openfast output into dataframe for each turbine in farm
        turbine_dfs = [_load_single_openfast(filepath, frequency=1, columns=columns) for filepath in filepaths]

        # Join into single dataframe using the Time column
        farm = turbine_dfs[0]
        for turbine in turbine_dfs[1:]:
            farm = pd.merge(farm, turbine, how='inner', on='Time')

        return farm


def multi_load(filepaths, frequency, config, cpus=12, columns=None):
    """
    Parallel load for OpenFAST simulation data.
    Given a list of filepaths to OpenFAST outputs, create dataframes with 8 sensor values at a given frequency.
    If we are loading farm simulations we will create a dataframe of each farm run.
    Likewise file paths for farm simulations should be nested lists.
    A list of individual turbine outputs per each farm inside a list of farms (i.e. call "get_output_files" in
    a list generator)

    :param filepaths: absolute filepaths to OpenFAST outputs
    :param frequency: sampling frequency in seconds
    :param config: 'single' or 'farm
    :param cpus: number of cpus to use for the load
    :param columns: which sensor values to load
    :return: list of dataframes, each holding the values of a simulation's OpenFAST data
    """

    with Pool(processes=cpus) as pool:
        results = [pool.apply_async(load_simulation, args=(outfile, frequency, config, columns)) for outfile in filepaths]

        dfs = [r.get() for r in results]

    return dfs


def build_cache(data, data_dir):
    """
    Saves a dataset of n sequences to a np.memmap on disk
    :param data: np.array (n_windows, timesteps, features)
    :param data_dir: absolute path to data directory
    :return: None
    """
    # Write shape to pickle
    with open(data_dir + "shape.pkl", "wb") as f:
        pickle.dump(data.shape, f)

    fp = np.memmap(data_dir + "cache.dat", dtype='float32', mode='w+', shape=data.shape)
    fp[:] = data[:]
    del fp


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


def run_openfast_sim(simulation_dir, windspeed, turbulence_class, turbsim_template, inflow_baseline, openfast_fst):
    """
    Run a full OpenFAST simulation, using the given template files
    :param simulation_dir: directory to put run output
    :param windspeed: windspeed (m/s) for simulation
    :param turbulence_class: IEC turbulence class (A, B, or C)
    :param turbsim_template: template ".inp" file for Turbsim
    :param inflow_baseline: InflowWind v3.01.* INPUT FILE (".dat"); required for OpenFAST
    :param openfast_fst: OpenFAST example INPUT FILE (".fst"); required for OpenFAST

    :return: Success of simulatiion
    """
    # Random seed for turbulence
    seed = np.random.randint(-2147483648, 2147483648)

    # Copy in input files
    shutil.copy(inflow_baseline, simulation_dir)
    shutil.copy(openfast_fst, simulation_dir)
    _write_turbsim_imp(simulation_dir, turbsim_template, seed, windspeed, turbulence_class)

    subprocess.run(["turbsim", simulation_dir + turbsim_template])
    status = subprocess.run(["openfast", simulation_dir, openfast_fst])
    return status.returncode


def iec_class_to_float(iec_class):
    """
    Converts IEC turbulence characteristic (A B or C) to expected hub-height turbulence intensity (float value)
    :param iec_class: IEC turbulence characteristic
    :return: hub-height turbulence intensity; float
    """
    if iec_class == 'A':
        return 0.16
    elif iec_class == 'B':
        return 0.14
    elif iec_class == 'C':
        return 0.12
    else:
        raise ValueError('Valid iec_class not in ["A", "B" or "C"]')


def run_floris_sim(farm_json, wind_direction, wind_speed, turbulence_class):
    """
    Runs a FLORIS simulation on a provided layout with a given wind direction, speed, and turbulence class.
    :param farm_json: Location of floris .json file with the farm layout and turbine spec
    :param wind_direction: direction of incoming wind
    :param wind_speed: global speed of incoming wind
    :param turbulence_class: IEC turbulence class
    :return: dictionary containing lists of windspeeds and turbulence intensities for all turbines in farm after
            wake is calculated

    """
    # Read FLORIS json template and assign new values
    with open(farm_json) as f:
        floris_config = json.load(f)
        floris_config['farm']['properties']['wind_speed'] = wind_speed
        floris_config['farm']['properties']['wind_direction'] = wind_direction
        floris_config['farm']['properties']['turbulence_intensity'] = iec_class_to_float(turbulence_class)

    # Create FLORIS model
    floris_model = floris.Floris(io.BytesIO(floris_config))

    # Calculate wake BEFORE getting turbine velocities
    floris_model.farm.flow_field.calculate_wake()

    turbine_speeds = [turb.average_velocity for turb in floris_model.farm.turbines]
    turbine_intensities = [turb.turbulence_intensity for turb in floris_model.farm.turbines]

    return {'turbine_speeds': turbine_speeds, 'turbine_intensities': turbine_intensities}


def _write_turbsim_imp(turbine_dir, turbsim_template, seed, wind_speed, turb_intensity):
    """
    Writes Turbsim ".inp" to destination using template with specified random seed, wind speed
    and turbulence inensity

    :param turbine_dir: absolute path to write ".inp"
    :param turbsim_template: template ".inp" file for Turbsim
    :param seed: random seed
    :param wind_speed: wind speed
    :param turb_intensity: turbulence intensity
    :return: None
    """
    with open(turbsim_template, "r") as turbsim_input:
        # Set the random seed
        new_text = re.sub('(TURBULENCE_SEED)', str(seed), turbsim_input.read())
        # Set the wind speed
        new_text = re.sub('WIND_SPEED', '%.3f' % wind_speed, new_text)
        # Set turb intensity
        new_text = re.sub('TURBULENCE_CLASS', turb_intensity, new_text)

    # Write to output
    output_file = turbine_dir + "/90m_12mps_twr.inp"
    with open(output_file, "w+") as tsim_output:
        tsim_output.write(new_text)


def stats_turbine(turbine_dir):
    """
    Get the wind speed and average power produced from a single turbine

    :param turbine_dir: path containing turbine's Turbsim ".inp" file as well as the
                        OpenFAST ".out" file
    :return: windspeed, mean and std power
    """
    inp_file = turbine_dir + "/90m_12mps_twr.inp"
    fst_out = turbine_dir + "/5MW_Land_DLL_WTurb.out"

    # Get windspeed from turbsim input
    with open(inp_file, "r") as f:
        lines = f.readlines()

        for line in lines:
            if re.search(r'URef', line):
                windspeed = float(line[1:7])
                break

    # Get mean and sddev gen power from openfast output
    scada_df = load_simulation(filepaths=fst_out, frequency=1, config='single')

    if scada_df is None:
        return None

    pwr_mean = scada_df['GenPwr'].mean()
    pwr_err = scada_df['GenPwr'].std()

    return windspeed, pwr_mean, pwr_err


def get_nrel_pwr_curve(nrel_performance_file):
    """
    Get a power curve from a NREL turbine performace .txt file

    :param nrel_performance_file: absolute path
    :return: dataframe with windspeeds and generated power
    """

    nrel_df = pd.read_csv(nrel_performance_file, sep='\t', encoding='latin1')
    nrel_df = nrel_df.drop(0)
    windspeeds = nrel_df['WindVxi'].astype('int').to_numpy()
    gen_pwr = nrel_df['GenPwr'].astype('float32').to_numpy()

    return pd.DataFrame({'windspeed': windspeeds, 'gen_pwr': gen_pwr})


def get_generated_power_curve(data_dir):
    """
    Get average power and windspeed for each simulation in the data directory

    :param data_dir: directory containing all simulation runs with OpenFAST output
    :return: dataframe of windspeeds and their corresponding mean generated power stats
    """
    windspeeds = []
    pwr_means = []
    pwr_errs = []
    with Pool(processes=10) as pool:

        paths = [data_dir + run for run in os.listdir(data_dir)]
        print("Getting Curves %s to %s ...." % (paths[0], paths[-1]))
        results = [pool.apply_async(stats_turbine, args=(path,)) for path in paths]
        print("Waiting for results...")

        for idx, r in enumerate(results):
            print_progress_bar(idx+1, len(paths))
            sys.stdout.flush()
            stats = r.get()
            if stats is not None:
                windspeed, mean, err = stats
                windspeeds.append(windspeed)
                pwr_means.append(mean)
                pwr_errs.append(err)
    print(windspeeds, pwr_means)
    return pd.DataFrame({'windspeed': windspeeds, 'pwr_means': pwr_means, 'pwr_std': pwr_errs})


def load_openfast_outputs(data_dir, sampling_freq, window_len, window_step, config, cols=None):
    """
    Crawls a directory of OpenFAST simulation run outputs (each with its own subdir) loads data with a given
    sampling frequency and creates sliding windows for training/testing
    :param data_dir: parent directory containing OpenFAST runs
    :param sampling_freq: frequency in seconds to sample OpenFAST outputs
    :param window_len: length in seconds of sliding windows
    :param window_step: step increment of sliding windows (e.g. a 600 second OpenFAST run
                        with 1 second sampling_freq and 60 second window_len will
                        give 520 windows with 1 seconds window_step)
    :param config: single turbine or farm
    :param cols: columns of OpenFAST data to sample collect (in None, defaults to
                        ['Time', 'Wind1VelX', 'Wind1VelY', 'Wind1VelZ', 'BldPitch1',
                         'NacYaw', 'RotSpeed', 'GenSpeed', 'GenPwr'])
    :return: np.array stack of SCADA windows (n_windows, timesteps, n_sensors)
    """

    # Get list of OpenFAST outputs files in data_dir
    if config == 'single':
        openfast_outfiles = get_output_files(data_dir)

    # For each farm simulation in data_dir, get list of OpenFAST outputs
    else:
        sim_dirs = [data_dir + "/" + sim for sim in os.listdir(data_dir)]
        openfast_outfiles = [get_output_files(sim) for sim in sim_dirs]

    print("Found {} files: {} .....".format(len(openfast_outfiles), openfast_outfiles[:3]), "\n")

    print("Loading simulations")
    scada_dfs = multi_load(openfast_outfiles, frequency=sampling_freq, config=config, columns=cols)
    sys.stdout.flush()

    print("Transforming DFs")
    scada_windows = np.array([create_windows(df, window_len=window_len, step=window_step)
                              for df in scada_dfs if len(df) == 601])

    # Stack all windows and normalize by sensor
    return np.vstack(scada_windows)


def print_progress_bar(idx, total):
    """
    Utility function that prints a progress bar to be called in loop iterations
    :param idx: current index
    :param total: final index
    :return:
    """
    bar_len = 50
    filled_len = int(bar_len * idx // total)
    fill = '█'
    bar = fill * filled_len + '-' * (bar_len - filled_len)
    print('\r Complete |%s| %d of %d  '
          % (bar, idx, total), end='\r')
    if idx == total:
        print()


