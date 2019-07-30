from .gan import Seq_GAN
import sherpa
from sklearn.model_selection import train_test_split
import numpy as np


class GAN_Herd:
    """
    Class build for hyper parameter optimization of Seq_GAN
    """

    def __init__(self, pop, log_dir):
        parameters = [
            # Auto Encoder Parameters
            sherpa.Discrete('hidden_units_AE', range=[2, 8]),
            sherpa.Discrete('lstm_units_AE', range=[2, 16]),
            sherpa.Continuous('dropout_AE', range=[0, 0.5]),
            sherpa.Continuous('lr_AE', range=[0.000001, 0.1], scale='log'),
            # Detector Parameters
            sherpa.Discrete('hidden_units_DET', range=[2, 8]),
            sherpa.Discrete('lstm_units_DET', range=[2, 8]),
            sherpa.Continuous('dropout_DET', range=[0, 0.5]),
            sherpa.Continuous('leaky_alpha_DET', range=[0.01, 0.4]),
            sherpa.Continuous('lr_DET', range=[0.000001, 0.1], scale='log'),
            # GAN parameters
            sherpa.Continuous('lr_GAN', range=[0.000001, 0.1], scale='log')]

        # Set an evolutionary algorithm for parameter search, enforce early stopping
        algorithm = sherpa.algorithms.PopulationBasedTraining(population_size=pop)
        rule = sherpa.algorithms.MedianStoppingRule(min_iterations=5, min_trials=1)
        self.study = sherpa.Study(parameters, algorithm, lower_is_better=True, stopping_rule=rule, dashboard_port=9800)

        self.logs_dir = log_dir

    def study_the_population(self, data, n_epochs, batch_size, cost_optimized):

        for tr_idx, trial in enumerate(self.study):

            # Create model for each trial (fix latent dimension size)
            madcow = MADCOW_AE_GAN(parameters=trial.parameters, cost_optimized=cost_optimized, batch_size=batch_size,
                                   data_shape={'timesteps': data.shape[1], 'features': data.shape[2], 'latent_dim': 1})

            # Set validation data (will not change through training
            training_windows, validation_windows = train_test_split(data, test_size=0.05)
            madcow.load_validation_data(validation_windows)

            for e_idx in range(n_epochs):
                # Sample and load
                epoch_windows = training_windows[np.random.choice(training_windows.shape[0], 100000, replace=False), :]
                madcow.load_training_data(epoch_windows)

                # Train and get losses
                train_loss, val_loss, metrics = madcow.train_epoch(epoch_id=e_idx, batch_size=batch_size)
                print(train_loss, val_loss, metrics)

                self.study.add_observation(trial=trial,
                                           iteration=e_idx,
                                           objective=train_loss['combined'],
                                           context={'0_det_val_loss': val_loss['det'],
                                                    '1_gan_val_loss': val_loss['gan'],
                                                    '2_ae_val_loss': val_loss['ae'],
                                                    'acc': metrics['acc'],
                                                    'f1_score': metrics['f1']})
                if self.study.should_trial_stop(trial):
                    break

            self.study.finalize(trial)
            print(self.study.results)
            # Save study
            self.study.save(self.logs_dir)



