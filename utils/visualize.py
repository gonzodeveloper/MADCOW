import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse


def parse_args():
    """
    Parse command line arguments (Only needed when plotting power curves directly
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        help="directory containing dataset cache")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="directory to log results")
    parser.add_argument("--nrel_pwr_file", type=str, required=True,
                        help="txt file with NREL 5MW power curve")
    parser.add_argument("--load_cached", action='store_true',
                        help="generated power curve is cached, load from log dir")
    return vars(parser.parse_args())


def plot_sensors(title_1, title_2, window_1, window_2, filepath=None):
    """
    Produces a comparative heatmap plot of two sensor windows.
    Can also be used to compare a window with its own reconstructions.
    Either produces an interactive plot or saves to file.

    :param title_1: title for first window
    :param title_2: title for second window
    :param window_1: first sensor window np.array shape=(timesteps, sensors)
    :param window_2: second sensor window np.array shape=(timesteps, sensors)
    :param filepath: if set this will define a save location for the plot
    :return:
    """

    # TEMP FOR 2 SENSOR
    labels = ["Wind1VelX", "Wind1VelY", "Wind1VelZ", "BldPitch1", "NacYaw", "RotSpeed", "GenSpeed", "GenPwr"]
    # labels = ["RotSpeed", "GenPwr"]
    fig, axes = plt.subplots(nrows=len(labels), ncols=2)

    sensor_winodw = np.rot90(window_1)
    compare_window = np.rot90(window_2)

    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0, 0].set_title(title_1, fontsize=14)

    axes[0, 1].set_title(title_2, fontsize=14)

    _min, _max = np.amin(np.vstack((window_1, window_2))), np.amax(np.vstack((window_1, window_2)))

    for idx, (ax, name) in enumerate(zip(axes[::, 0], labels)):

        gradient = np.vstack((sensor_winodw[idx], sensor_winodw[idx]))
        ax.imshow(gradient, aspect='auto', cmap='gist_heat')
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3] / 2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    for idx, ax in enumerate(axes[::, 1]):
        gradient = np.vstack((compare_window[idx], compare_window[idx]))
        ax.imshow(gradient, aspect='auto', cmap='gist_heat')

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes[::, 0]:
        ax.set_axis_off()
    for ax in axes[::, 1]:
        ax.set_axis_off()

    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
        plt.close(fig)


def plot_reconstruction_errors(labels, reconstruction_errors, filepath):

    plt.title('Reconstruction Errors')
    plt.xlabel('Similarity Error')
    plt.ylabel('Type (0 = normal, 1 = anomalous)')
    plt.scatter(reconstruction_errors, labels)
    plt.savefig(filepath)


def plot_threshold_f1(thesholds, f1_scores, filepath):
    plt.title('F1 Scores by Decision Threshold')
    plt.xlabel('Decision threshold')
    plt.ylabel('F1 Score')
    plt.scatter(thesholds, f1_scores)
    plt.savefig(filepath)


if __name__ == '__main__':
    '''
    Call directly with args to plot power curve
    '''

    from etl import get_nrel_pwr_curve, get_generated_power_curve

    args = parse_args()

    if not args['load_cached']:
        # Get power curves openfast results and save
        openfast_df = get_generated_power_curve(args['data_dir'])
        print(openfast_df)
        openfast_df.to_csv(args['log_dir'] + "openfast_pwr_curve.csv", index=False)

    else:
        # Load from cache
        openfast_df = pd.read_csv(args['log_dir'] + "openfast_pwr_curve.csv", engine='python')

    nrel_df = get_nrel_pwr_curve(args['nrel_pwr_file'])

    plt.title('OpenFAST vs NREL 5MW Power Curves')
    plt.xlabel('Windspeed (m/s)')
    plt.ylabel('Mean Generated Power(kW)')
    plt.grid()
    plt.scatter(openfast_df['windspeed'], openfast_df['pwr_means'], alpha=0.85, s=0.8, label="OpenFAST samples")
    plt.plot(nrel_df['windspeed'], nrel_df['gen_pwr'], c='black', linewidth=1, label="NREL 5MW measurements")

    plt.legend(loc='lower right')
    plt.savefig(args['log_dir'] + 'power_curve.png')



