from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, CuDNNLSTM, LeakyReLU, Input, Bidirectional, RepeatVector
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from keras.losses import binary_crossentropy
import keras.backend as K
# from tflearn.objectives import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from MADCOW.models.loss_layers import *


class Seq_GAN:
    """
    Hybrid Autoencoder GAN for anomaly detection.
    Can optionally be trained with AUPRC objecttive provided by the global_objectives library.
    """
    def __init__(self, model_load_dir=None, data_shape=None, parameters=None, cost_optimized=False, batch_size=None):
        """
        Initialize the model from parameter dictionary or load existing model
        :param model_load_dir: directory containing saved models
        :param data_shape: dictionary with shape of training data
                            'latent_dim': desired size of the latent sampling space (int)
                            'timesteps': the length of the time series data window  (int)
                            'features': number of features we are training  (int)
        :param parameters: dictionary specifying architecture and hyperparameters of the MADCOW
                            'lstm_units_AE':
                            'hidden_units_AE':
                            'dropout_AE':
                            'lr_AE':
                            'lstm_units_DET':
                            'hidden_units_AE':
                            'dropout_AE':

        :param cost_optimized:
        :param batch_size:
        """

        if model_load_dir is None:

            if data_shape is None:
                raise RuntimeError('If model is not loaded from directory, requires data shape')
            if parameters is None:
                raise RuntimeError('If model is not loaded from directory, requires parameters ')

            # Set dimension for latent space Z
            self.latent_dim = data_shape['latent_dim']
            self.features = data_shape['features']
            self.timesteps = data_shape['timesteps']

            # Cost optimized model require a fixed batch size
            self.cost_optimized = cost_optimized
            self.batch_size = batch_size

            # Build base NNs
            self.encoder = self.build_encoder(lstm_units=parameters['lstm_units_AE'],
                                              hidden_units=parameters['hidden_units_AE'],
                                              dropout=parameters['dropout_AE'])
            self.decoder = self.build_decoder(lstm_units=parameters['lstm_units_AE'],
                                              hidden_units=parameters['hidden_units_AE'],
                                              dropout=parameters['dropout_AE'])
            self.detector = self.build_detector(lstm_units=parameters['lstm_units_DET'],
                                                hidden_units=parameters['hidden_units_DET'],
                                                dropout=parameters['dropout_DET'],
                                                alpha=parameters['leaky_alpha_DET'],
                                                lr=parameters['lr_DET'])

            # Build Composite NNs
            self.GAN = self.create_gan(lr=parameters['lr_GAN'])
            self.AE = self.create_autoencoder(lr=parameters['lr_AE'])
        else:
            self.load_models(model_load_dir)
            # Unnecessary if we've already constructed models
            self.latent_dim = None
            self.timesteps = None
            self.features = None

        # Load data ad hoc
        self.training_data = None
        self.validation_data = None

        # Cost Optimized Loss function needs fixed batch_size
        self.BATCH_SIZE = None

        # Tensor Boards
        self.tensorboards = dict()

    def save_models(self, save_dir):
        self.encoder.save(save_dir + 'encoder.h5')
        self.decoder.save(save_dir + 'decoder.h5')
        self.detector.save(save_dir + 'detector.h5')
        self.GAN.save(save_dir + 'GAN.h5')
        self.AE.save(save_dir + 'AE.h5')

    def load_models(self, save_dir):
        self.encoder = load_model(save_dir + 'encoder.h5')
        self.decoder = load_model(save_dir + 'decoder.h5')
        self.detector = load_model(save_dir + 'detector.h5', compile=False)
        self.GAN = load_model(save_dir + 'GAN.h5', compile=False)
        self.AE = load_model(save_dir + 'AE.h5', compile=False)

    def load_training_data(self, data, validation_split=True):
        if validation_split:
            self.training_data, self.validation_data = train_test_split(data, test_size=0.1)
        else:
            self.training_data = data

    def load_validation_data(self, data):
        self.validation_data = data

        # Stack validation data into single batch
        # self.validation_data = np.vstack(self.validation_data)

    def build_encoder(self, lstm_units, hidden_units, dropout):
        encoder = Sequential()
        # First Forward Feed CuDNNLSTM
        encoder.add(CuDNNLSTM(input_shape=(self.timesteps, self.features),
                              units=lstm_units,
                              kernel_initializer='glorot_normal',
                              recurrent_initializer='glorot_normal',
                              bias_initializer='zeros',
                              return_sequences=True))
        encoder.add(Dropout(dropout))

        encoder.add(CuDNNLSTM(units=hidden_units,
                              kernel_initializer='glorot_normal',
                              recurrent_initializer='glorot_normal',
                              bias_initializer='zeros',
                              return_sequences=True))
        encoder.add(Dropout(dropout))

        encoder.add(CuDNNLSTM(units=self.latent_dim,
                              kernel_initializer='glorot_normal',
                              recurrent_initializer='glorot_normal',
                              bias_initializer='zeros',
                              return_sequences=False))

        # For GPU leverage
        encoder = multi_gpu_model(encoder, gpus=2)
        return encoder

    def build_decoder(self, lstm_units, hidden_units, dropout):
        decoder = Sequential()
        decoder.add(RepeatVector(self.timesteps, input_shape=(self.latent_dim,)))
        # Decoder portion of network
        decoder.add(CuDNNLSTM(units=hidden_units,
                              kernel_initializer='glorot_normal',
                              recurrent_initializer='glorot_normal',
                              bias_initializer='zeros',
                              return_sequences=True))
        decoder.add(Dropout(dropout))
        decoder.add(CuDNNLSTM(units=lstm_units,
                              kernel_initializer='glorot_normal',
                              recurrent_initializer='glorot_normal',
                              bias_initializer='zeros',
                              return_sequences=True))
        decoder.add(Dropout(dropout))
        decoder.add(CuDNNLSTM(units=self.features,
                              kernel_initializer='glorot_normal',
                              recurrent_initializer='glorot_normal',
                              bias_initializer='zeros',
                              return_sequences=True))

        # For GPU leverage
        decoder = multi_gpu_model(decoder, gpus=2)
        return decoder

    def build_detector(self, lstm_units, hidden_units, dropout, lr, alpha):
        detector = Sequential()
        # Dual Bidirectional LSTMs to check fidelity of entire sequences
        detector.add(Bidirectional(
            CuDNNLSTM(units=lstm_units,
                      kernel_initializer='glorot_normal',
                      recurrent_initializer='glorot_normal',
                      bias_initializer='zeros',
                      return_sequences=True), input_shape=(self.timesteps, self.features)))
        detector.add(Dropout(dropout))

        detector.add(Bidirectional(
            CuDNNLSTM(units=lstm_units,
                      kernel_initializer='glorot_normal',
                      recurrent_initializer='glorot_normal',
                      bias_initializer='zeros')))
        detector.add(Dropout(dropout))
        # Dense Layers !!!
        detector.add(Dense(units=hidden_units,
                           kernel_initializer='he_normal',
                           bias_initializer='zeros'))
        detector.add(LeakyReLU(alpha))
        detector.add(Dropout(dropout))
        detector.add(Dense(units=int(hidden_units / 2),
                           kernel_initializer='he_normal',
                           bias_initializer='zeros'))
        detector.add(LeakyReLU(alpha))
        detector.add(Dropout(dropout))
        detector.add(Dense(1, activation='sigmoid'))

        # For GPU leverage
        detector = multi_gpu_model(detector, gpus=2)
        adam = Adam(lr)

        bat_len = self.batch_size

        def keras_prc_loss(y_true, y_pred, batch_size=bat_len):
            y_true = K.reshape(y_true, (batch_size, 1))
            y_pred = K.reshape(y_pred, (batch_size, 1))
            get_num_labels = lambda labels: 1
            return precision_recall_auc_loss(y_true, y_pred)[0]

        objective = binary_crossentropy if not self.cost_optimized else keras_prc_loss
        detector.compile(optimizer=adam, loss=objective)
        return detector

    def create_gan(self, lr):
        # An random seq from z-dim latent space
        inputs = Input(shape=(self.latent_dim, ))
        # Generator creates a sequence
        gen_seq = self.decoder(inputs)
        # Discriminator evaluates the sequence
        evaluation = self.detector(gen_seq)

        # Build model
        gan = Model(inputs=inputs, outputs=evaluation)
        # Leverage GPUs
        gan = multi_gpu_model(gan, gpus=2)
        self.detector.trainable = False
        adam = Adam(lr)
        gan.compile(optimizer=adam, loss='binary_crossentropy')

        print()
        print("-----GAN Summary------")
        gan.summary()
        return gan

    def create_autoencoder(self, lr):
        # Input a full sequence
        inputs = Input(shape=(self.timesteps, self.features))
        # Generator creates a sequence
        latent_mapping = self.encoder(inputs)
        # Decoder reconstructs the sequence from Z space
        reconstruction = self.decoder(latent_mapping)

        # Build model
        ae = Model(inputs=inputs, outputs=reconstruction)

        # Leverage GPUs
        ae = multi_gpu_model(ae, gpus=2)
        # Freeze decoder (gets trained by GAN)
        self.decoder.trainable = False
        adam = Adam(lr)
        ae.compile(optimizer=adam, loss='binary_crossentropy')

        print()
        print("-----AutoEncoder Summary------")
        ae.summary()
        return ae

    def set_up_tensorboards(self, logs_dir, batch_size):
        # Instantiate Boards
        self.tensorboards['det'] = TensorBoard(log_dir=logs_dir + "det/",
                                               histogram_freq=0, batch_size=batch_size, write_grads=True)
        self.tensorboards['det'].set_model(self.detector)

        self.tensorboards['gan'] = TensorBoard(log_dir=logs_dir + "gan/",
                                               histogram_freq=0, batch_size=batch_size, write_grads=True)
        self.tensorboards['gan'].set_model(self.GAN)

        self.tensorboards['ae'] = TensorBoard(log_dir=logs_dir + "ae/",
                                              histogram_freq=0, batch_size=batch_size, write_grads=True)
        self.tensorboards['ae'].set_model(self.AE)

    def report_batch_to_tensorboard(self, batch_id, det_loss, gan_loss, ae_loss):
        # Only report if we have tensorboards
        if self.tensorboards:
            self.tensorboards['det'].on_batch_end(batch_id, logs={'loss': det_loss})
            self.tensorboards['gan'].on_batch_end(batch_id, logs={'loss': gan_loss})
            self.tensorboards['ae'].on_batch_end(batch_id, logs={'loss': ae_loss})

    def report_epoch_to_tensorboard(self, epoch_id, training_loss, val_loss):
        # Only report if we have tensorboards
        if self.tensorboards:
            self.tensorboards['det'].on_epoch_end(epoch_id,
                                                  logs={'loss': training_loss['det'], 'val_loss': val_loss['det']})
            self.tensorboards['gan'].on_epoch_end(epoch_id,
                                                  logs={'loss': training_loss['gan'], 'val_loss': val_loss['gan']})
            self.tensorboards['ae'].on_epoch_end(epoch_id,
                                                 logs={'loss': training_loss['ae'], 'val_loss': val_loss['ae']})

    def train_epoch(self, epoch_id, batch_size):

        if self.training_data is None:
            raise RuntimeError("Must load training data before training...")

        print("Epoch {}".format(epoch_id))

        self.BATCH_SIZE = batch_size

        # Shuffle data
        np.random.shuffle(self.training_data)

        # Legacy for non-batched training data
        num_batches = len(self.training_data) / batch_size
        seq_batches = np.split(self.training_data, num_batches)

        training_losses = {'det': [], 'gan': [], 'ae': []}

        for idx, real_seqs in enumerate(seq_batches):

            batch_size = real_seqs.shape[0]

            # Make labels
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Get generated/decoded sequences
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_seqs = self.decoder.predict(noise)

            # Train Discriminator/Detector First
            d_loss_real = self.detector.train_on_batch(real_seqs, valid)
            d_loss_fake = self.detector.train_on_batch(gen_seqs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator/Decoder Next
            g_loss = self.GAN.train_on_batch(noise, valid)

            # Finally Train AutoEncoder
            a_loss = self.AE.train_on_batch(real_seqs, real_seqs)

            # Report losses
            self.report_batch_to_tensorboard(batch_id=idx, det_loss=d_loss, gan_loss=g_loss, ae_loss=a_loss)

            # Save for SHERPA
            training_losses['det'].append(d_loss)
            training_losses['gan'].append(g_loss)
            training_losses['ae'].append(a_loss)
            # Print for my eyes
            Seq_GAN.print_batch(idx+1, len(seq_batches), d_loss, g_loss, a_loss)

        # Average and combined training losses
        for idx, all_loss in training_losses.items():
            training_losses[idx] = np.mean(all_loss)

        # Combined training loss will be our objective function for SHERPA
        training_losses['combined'] = np.sum(list(training_losses.values()))

        # Validate data anf get reports
        val_losses, metrics = self.validate_epoch()

        # TensorBoard Reports
        self.report_epoch_to_tensorboard(epoch_id, training_losses, val_losses)

        # Send back figures for SHERPA
        return training_losses, val_losses, metrics

    def validate_epoch(self):
        # Metrics and val loss for detector (Use 90/10 real vs generated)
        val = self.validation_data
        val_samples = len(self.validation_data)

        # Get generated/decoded sequences
        noise = np.random.normal(0, 1, (val_samples, self.latent_dim))
        gen = self.decoder.predict(noise)

        valid = np.ones((val_samples, 1))
        fake = np.zeros((val_samples, 1))

        x = np.vstack((val, gen))
        y = np.vstack((valid, fake))

        y_pred = np.rint(self.detector.predict(x))
        det_loss = self.detector.test_on_batch(x, y) if not self.cost_optimized else 0
        det_acc = accuracy_score(y, y_pred)
        det_f1 = f1_score(y, y_pred)

        # Loss metric for GAN
        gan_loss = self.GAN.test_on_batch(noise, valid)

        # Loss of AE
        ae_loss = self.AE.test_on_batch(val, val)

        # Give back reports on validation
        return {'det': det_loss, 'gan': gan_loss, 'ae': ae_loss}, {'f1': det_f1, 'acc': det_acc}

    def predict(self, windows):
        detections = self.detector.predict(windows)
        reconstructions = self.AE.predict(windows)

        return detections, reconstructions

    @staticmethod
    def print_batch(idx, total, d_loss, g_loss, a_loss):
        bar_len = 50
        filled_len = int(bar_len * idx // total)
        fill = '█'
        bar = fill * filled_len + '-' * (bar_len - filled_len)
        print('\r Batch %d of %d: |%s| Discriminator Loss: %f   |  GAN Loss: %f '
              % (idx, total, bar, d_loss, g_loss), end='\r')
        if idx == total:
            print()

    @staticmethod
    def report_epoch(acc, f1, auroc):
        print('COMPLETE!    Validation Accuracy: %f  F-Score: %f  AUROC: %f' % (acc, f1, auroc))
        print()



