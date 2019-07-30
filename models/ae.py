from keras.models import Model, Sequential, load_model
from keras.layers import Dense, CuDNNLSTM, RepeatVector, TimeDistributed, Dropout, Input, \
    Lambda, Conv2D, UpSampling2D, MaxPooling2D, Cropping2D, Flatten, Reshape
from keras.losses import mse
import keras.backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np


class Seq_Autoencoder:
    """
    Base class of sequential autoencoder. Basically a simple wrapper around Keras methods.
    """

    def __init__(self, log_dir, load_model=False):
        """
        Either instanciates a new autoencoder with its own log directory, or loads one from the log directory
        :param log_dir: directory that stores model .h5 file and Tensorflow logs
        :param load_model: loads a model from file
        """
        # Used for loading and saving model as well as storing tensorflow and sherpa outputs
        self.log_dir = log_dir
        if load_model is False:
            self.autoencoder = None
        else:
            self.load_model()

        # Load data ad hoc
        self.training_data = None
        self.validation_data = None

    def save_model(self):
        """
        Saves the model to class log_dir
        :return:
        """
        self.autoencoder.save(self.log_dir + 'model.h5')

    def load_model(self):
        """
        Loads a model.h5 from class log_dir
        :return:
        """
        self.autoencoder = load_model(self.log_dir + 'model.h5')

    def load_training_data(self, data, validation_split=False, frac_val=0.1):
        """
        Loads training data and performs a validation split if prompted
        :param data: np.array (n_sequences, timesteps, features)
        :param validation_split: whether or not to perform a validation split
        :param frac_val: fraction of data to leave out for validation
        :return:
        """
        if validation_split:
            self.training_data, self.validation_data = train_test_split(data, test_size=frac_val)
        else:
            self.training_data = data

    def load_validation_data(self, data):
        """
        Separately load validation data
        :param data: np.array (n_sequences, timesteps, features)
        :return:
        """
        self.validation_data = data

    def build_model(self, ae_type, timesteps, features, filter_1=None, filter_2=None, filter_3=None,
                    lstm_units=None, hidden_units=None, latent_dim=None,
                    dropout=None, batch_size=None, epsilon_std=1):
        """
        Creates either a variational or de-noising auto encoder as our base model.
        Note that de-noising requires a dropout rate, vae requires a set batch size and epsilon std.
        :param ae_type: either 'de-noising' or 'variational'
        :param timesteps: number of timesteps in sequence windows for training
        :param filter_1
        :param filter_2
        :param filter_3
        :param features: number of features in training sequences
        :param lstm_units: number of lstm units in first and last layers
        :param hidden_units: number of lstm units in hidden layers
        :param latent_dim: size of the latent dimension
        :param dropout: dropout rate (for de-noising)
        :param epsilon_std: scalar for standard deviation in VAE
        :param batch_size: fixed batch size for reparameterization in VAE
        :return:
        """

        if ae_type == 'de-noising':
            self.autoencoder = self._build_denoiser(timesteps, features, lstm_units, hidden_units,
                                                    latent_dim, dropout)

        elif ae_type == 'variational':
            self.autoencoder = self._build_vae(timesteps, features, lstm_units, hidden_units,
                                               latent_dim, batch_size, epsilon_std)
        elif ae_type == 'conv':
            self.autoencoder = self._build_conv_ae(timesteps, features, filter_1, filter_2, filter_3)
        elif ae_type == 'dense':
            self.autoencoder = self._build_dense(timesteps, features)
        else:
            raise ValueError("ae_type not in ['de-noising', 'varriational']")

        print("-----AutoEncoder Summary------")
        self.autoencoder.summary()

    def _build_denoiser(self, timesteps, features, lstm_units, hidden_units, latent_dim, dropout):
        """
        Builds a de-noising sequence autoencoder as the primary model
        :param features: number of features in training sequences
        :param lstm_units: number of lstm units in first and last layers
        :param hidden_units: number of lstm units in hidden layers
        :param latent_dim: size of the latent dimension
        :param dropout: dropout rate
        :return: compiled model
        """
        model = Sequential()
        # First Forward Feed CuDNNLSTM
        model.add(CuDNNLSTM(input_shape=(timesteps, features),
                            units=lstm_units,
                            kernel_initializer='glorot_normal',
                            recurrent_initializer='glorot_normal',
                            bias_initializer='zeros',
                            return_sequences=True))
        model.add(Dropout(rate=dropout))

        model.add(CuDNNLSTM(units=hidden_units,
                            kernel_initializer='glorot_normal',
                            recurrent_initializer='glorot_normal',
                            bias_initializer='zeros',
                            return_sequences=True))
        model.add(Dropout(rate=dropout))

        model.add(CuDNNLSTM(units=latent_dim,
                            kernel_initializer='glorot_normal',
                            recurrent_initializer='glorot_normal',
                            bias_initializer='zeros',
                            return_sequences=False))

        model.add(RepeatVector(timesteps))

        model.add(CuDNNLSTM(units=hidden_units,
                            kernel_initializer='glorot_normal',
                            recurrent_initializer='glorot_normal',
                            bias_initializer='zeros',
                            return_sequences=True))
        model.add(Dropout(rate=dropout))

        model.add(CuDNNLSTM(units=hidden_units,
                            kernel_initializer='glorot_normal',
                            recurrent_initializer='glorot_normal',
                            bias_initializer='zeros',
                            return_sequences=True))
        model.add(Dropout(rate=dropout))

        model.add(CuDNNLSTM(units=lstm_units,
                            kernel_initializer='glorot_normal',
                            recurrent_initializer='glorot_normal',
                            bias_initializer='zeros',
                            return_sequences=True))
        model.add(TimeDistributed(Dense(features)))

        # For GPU leverage
        model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def _build_vae(self, timesteps, features, lstm_units, hidden_units, latent_dim, batch_size, epsilon_std):
        """
        Builds a variational autoencoder as the primary model
        :param features: number of features in training sequences
        :param lstm_units: number of lstm units in first and last layers
        :param hidden_units: number of lstm units in hidden layers
        :param latent_dim: size of the latent dimension
        :param batch_size: fixed batch size required for reparameterization
        :param epsilon_std: scalar for standard deviation
        :return: compiled model
        """
        x = Input(shape=(timesteps, features,))

        # LSTM encoding
        h = CuDNNLSTM(units=lstm_units,
                      kernel_initializer='glorot_normal',
                      recurrent_initializer='glorot_normal',
                      bias_initializer='zeros',
                      return_sequences=True)(x)
        # LSTM encoding
        h = CuDNNLSTM(units=hidden_units,
                      kernel_initializer='glorot_normal',
                      recurrent_initializer='glorot_normal',
                      bias_initializer='zeros',
                      return_sequences=False)(h)

        # VAE Z layer
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                      mean=0., stddev=epsilon_std)
            return z_mean + z_log_sigma * epsilon

        z = Lambda(sampling, output_shape=(timesteps, latent_dim,))([z_mean, z_log_sigma])

        h = RepeatVector(timesteps)(z)
        h = CuDNNLSTM(units=hidden_units,
                      kernel_initializer='glorot_normal',
                      recurrent_initializer='glorot_normal',
                      bias_initializer='zeros',
                      return_sequences=True)(h)
        # LSTM encoding
        y = CuDNNLSTM(units=lstm_units,
                      kernel_initializer='glorot_normal',
                      recurrent_initializer='glorot_normal',
                      bias_initializer='zeros',
                      return_sequences=True)(h)

        vae = Model(x, y)

        def vae_loss(x, x_decoded_mean):
            xent_loss = mse(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return xent_loss + kl_loss

        # For GPU leverage
        model = multi_gpu_model(vae, gpus=2)
        model.compile(optimizer='adam', loss=vae_loss)
        return model

    def _build_conv_ae(self, timesteps, features, filters_1, filters_2, filters_3):

        model = Sequential()

        model.add(Conv2D(input_shape=(timesteps, features, 1),
                         filters=filters_1, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(input_shape=(timesteps, features, 1),
                         filters=filters_2, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Conv2D(input_shape=(timesteps, features, 1),
                         filters=filters_3, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D((2, 2), padding='same'))

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        model.add(Conv2D(input_shape=(timesteps, features, 1),
                         filters=filters_3, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(input_shape=(timesteps, features, 1),
                         filters=filters_2, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(input_shape=(timesteps, features, 1),
                         filters=filters_1, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
        model.add(Cropping2D((2, 0)))

        # For GPU leverage
        model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def _build_dense(self, timesteps, features):

        input_img = Input(shape=(timesteps,features, ))
        flat = Flatten()(input_img)
        encoded = Dense(256, activation='relu')(flat)
        encoded = Dense(128, activation='relu')(encoded)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(32, activation='relu')(encoded)

        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(256, activation='relu')(decoded)
        decoded = Dense(timesteps * features, activation='sigmoid')(decoded)
        formed = Reshape(target_shape=(timesteps, features,))(decoded)

        model = Model(input_img, formed)
        # For GPU leverage
        model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model


    def train_model(self, num_epochs, epoch_id, batch_size):
        if self.training_data is None:
            raise RuntimeError("Must load training data before training...")

        # Shuffle data
        np.random.shuffle(self.training_data)
        hist = self.autoencoder.fit(self.training_data, self.training_data,
                                    batch_size=batch_size,
                                    epochs=num_epochs,
                                    initial_epoch=epoch_id,
                                    validation_data=(self.validation_data, self.validation_data),
                                    callbacks=[TensorBoard(log_dir=self.log_dir, histogram_freq=1,
                                                           batch_size=batch_size, write_grads=True)])
        return hist.history

    def predict(self, data):
        return self.autoencoder.predict(data)



