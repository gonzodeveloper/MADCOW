import matplotlib.pyplot as plt
from pylab import gcf
import pandas as pd
import argparse


def parse_args():
    """
    Parse command line arguments
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True,
                        help="directory containing report csv and ")
    parser.add_argument("--report_name", type=str, required=True,
                        help="name of report csv to plot")
    parser.add_argument("--title", type=str,
                        help="title for interactive plot")
    return vars(parser.parse_args())


if __name__ == '__main__':

    args = parse_args()

    # Read report
    report_df = pd.read_csv(args['log_dir'] + args['report_name'])

    # Dataframe for each model
    knn_df = report_df[report_df['1_model_name'] == 'knn']
    pca_df = report_df[report_df['1_model_name'] == 'pca']
    cblof_df = report_df[report_df['1_model_name'] == 'cblof']
    iforest_df = report_df[report_df['1_model_name'] == 'iforest']
    ae_df = report_df[report_df['1_model_name'] == 'autoencoder']

    # Plotting
    figs, axes = plt.subplots(nrows=3, ncols=1)

    # Plot accuracies
    axes[0].set_xlim(0.5, 0.001)
    axes[0].set_xscale('log')
    axes[0].set_ylabel('Accuracy')
    axes[0].plot(knn_df['contamination'], knn_df['accuracy_score'], label='KNN')
    axes[0].plot(pca_df['contamination'], pca_df['accuracy_score'], label='PCA')
    axes[0].plot(cblof_df['contamination'], cblof_df['accuracy_score'], label='CBLOF')
    axes[0].plot(iforest_df['contamination'], iforest_df['accuracy_score'], label='IsolationForest')
    axes[0].plot(ae_df['contamination'], ae_df['accuracy_score'], label='Autoencoder')

    # Plot F_score
    axes[1].set_xlim(0.5, 0.001)
    axes[1].set_xscale('log')
    axes[1].set_ylabel('F-Score')
    axes[1].plot(knn_df['contamination'], knn_df['f1_score'], label='KNN')
    axes[1].plot(pca_df['contamination'], pca_df['f1_score'], label='PCA')
    axes[1].plot(cblof_df['contamination'], cblof_df['f1_score'], label='CBLOF')
    axes[1].plot(iforest_df['contamination'], iforest_df['f1_score'], label='IsolationForest')
    axes[1].plot(ae_df['contamination'], ae_df['f1_score'], label='Autoencoder')

    # Plot AUROC
    axes[2].set_xlim(0.5, 0.001)
    axes[2].set_xscale('log')
    axes[2].set_ylabel('AUROC')
    axes[2].plot(knn_df['contamination'], knn_df['AUROC'], label='KNN')
    axes[2].plot(pca_df['contamination'], pca_df['AUROC'], label='PCA')
    axes[2].plot(cblof_df['contamination'], cblof_df['AUROC'], label='CBLOF')
    axes[2].plot(iforest_df['contamination'], iforest_df['AUROC'], label='IsolationForest')
    axes[2].plot(ae_df['contamination'], ae_df['AUROC'], label='Autoencoder')

    # Make Legend Below
    box = axes[2].get_position()
    axes[2].set_position([box.x0, box.y0 + box.height * 0.1,
                               box.width, box.height * 0.9])

    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                      fancybox=True, shadow=True, ncol=5)

    fig = gcf()
    fig.suptitle(args['title'])
    plt.show()

