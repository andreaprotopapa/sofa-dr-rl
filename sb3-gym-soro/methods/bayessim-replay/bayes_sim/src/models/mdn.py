# Other imports
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# Creates a MDN with elliptical components (sigmas differ in each dimension)
# Imports of the Keras library parts we will need
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.initializers import lecun_normal
from tensorflow.keras.layers import Input, Dense, concatenate, Lambda
# Definition of the ELU+1 function
# With some margin to avoid problems of instability
from tensorflow.keras.layers import ELU
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from bayes_sim.src.models.random_features import RFF
from bayes_sim.src.models.random_features_tf import RFF as RFF_TF

import bayes_sim.src.utils.pdf as pdf
import wandb
from wandb.keras import WandbMetricsLogger
# Set up CPU or GPU
config = tf.compat.v1.ConfigProto(device_count={'CPU': 6, 'GPU': 2})
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class MDRFF(object):
    def __init__(self, ncomp=10, nfeat=154, inputd=None, cosOnly=False,
                 kernel="RBF", sigma=1, quasiRandom=True, outputd=None, weights=None, eps=0.):

        self.ncomp = ncomp  # number of mixture components
        self.nfeat = nfeat  # number of features
        self.inputd = inputd  # dimensionality of the input
        self.outputd = outputd  # dimensionality of the output
        self.quasiRandom = quasiRandom
        self.cosOnly = cosOnly
        self.sigma = sigma * np.ones(self.inputd)
        self.kernel = kernel
        self.rff = RFF(self.nfeat, self.inputd, self.sigma,
                       self.cosOnly, self.quasiRandom, self.kernel)
        self.weights = weights  # weight function
        self.scaler = StandardScaler()
        self.eps = eps

        # self.scaler = MinMaxScaler()
        # self.scaler = MaxAbsScaler()
        # self.scaler = Normalizer(norm='l2')
        # self.scaler = QuantileTransformer(output_distribution='normal', random_state=0)

        def elu_modif(x, a=1.):
            e = 1e-15
            return ELU(alpha=a)(x) + 1. + e

        # Note: The output size will be (outputd + 2) * ncomp

        # For reference, not used at the moment
        def log_sum_exp(x, axis=None):
            """Log-sum-exp trick implementation"""
            x_max = K.max(x, axis=axis, keepdims=True)
            return K.log(K.sum(K.exp(x - x_max),
                               axis=axis, keepdims=True)) + x_max

        def tril_matrix(elements):
            # tfd = tfp.distributions
            tril_m = tfp.math.fill_triangular(elements)
            tf.compat.v1.matrix_set_diag(tril_m, tf.add(tf.exp(tf.compat.v1.matrix_diag_part(tril_m)), self.eps))
            return tril_m

        def mean_log_Gaussian_like(y_true, parameters):
            # This version uses tensorflow_probability
            components = K.reshape(parameters, [-1, self.outputd +
                                                int(0.5 * (self.outputd + 1) *
                                                    self.outputd) + 1, self.ncomp])
            mu = components[:, :self.outputd, :]
            mu = K.reshape(mu, [-1, self.ncomp, self.outputd])
            sigma = components[:,
                    self.outputd: self.outputd + int(0.5 * (self.outputd + 1) * self.outputd), :]
            sigma = K.reshape(sigma, [-1, self.ncomp, int(0.5 * (self.outputd + 1) * self.outputd)])
            alpha = components[:, -1, :]

            # alpha = K.softmax(K.clip(alpha,1e-8,1.))
            alpha = K.clip(alpha, 1e-8, 1.)
            tfd = tfp.distributions
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=alpha),
                components_distribution=tfd.MultivariateNormalTriL(
                    loc=mu,
                    scale_tril=tril_matrix(sigma)))

            log_gauss = mix.log_prob(y_true)
            res = - K.mean(log_gauss)
            return res

        # This returns a tensor
        inputs = Input(shape=(self.nfeat,))

        # Note: Replace 'rmsprop' by 'adam' depending on your needs.
        FC_mus = Dense(units=self.outputd * self.ncomp,
                       activation='linear',
                       kernel_initializer='RandomUniform',
                       name='FC_mus')(inputs)
        FC_sigmas_d = Dense(units=self.outputd * self.ncomp,
                            activation='linear',
                            kernel_initializer='Ones',
                            kernel_regularizer=l2(10.),
                            name='FC_sigmas_d')(inputs)  # K.exp, W_regularizer=l2(1e-3)
        FC_sigmas = Dense(units=int(0.5 * (self.outputd - 1) * self.outputd * self.ncomp),
                          activation='linear',
                          kernel_initializer='Ones',
                          kernel_regularizer=l2(1.),
                          name='FC_sigmas')(inputs)  # K.exp, W_regularizer=l2(1e-3)
        FC_alphas = Dense(units=self.ncomp,
                          activation='softmax',
                          kernel_initializer='Ones',
                          name='FC_alphas')(inputs)

        output = concatenate([FC_mus, FC_sigmas_d, FC_sigmas, FC_alphas], axis=1)
        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile('adam', loss=mean_log_Gaussian_like)

    # Training function
    def train(self, x_data, y_data, nepoch=1000, plot=False, save=False, batch_size=100, wandb=None):
        lossHistory = LossHistory()

        # checkpoint
        if save:
            print("Training model....")
            filepath = "MDN--{epoch:02d}-{val_loss:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            start_time = datetime.now()
            # Fix nan values in the dataset
            ind = np.isnan(x_data)
            x_data[ind] = 0.
            x_dataS = self.scaler.fit_transform(x_data.T)
            x_feat = self.rff.toFeatures(x_dataS)
            w = None
            if self.weights is not None:
                w = self.weights.eval(x_data.T)
            callbacks = [lossHistory, checkpoint]
            if wandb is not None:
                callbacks.append(WandbMetricsLogger())
            self.model.fit(x_feat, y_data.T,
                           sample_weight=w,
                           epochs=nepoch,
                           validation_split=0.1,
                           batch_size=batch_size,
                           callbacks=callbacks)
        else:
            print("Training model....")
            start_time = datetime.now()
            # Fix nan values in the dataset
            ind = np.isnan(x_data)
            x_data[ind] = 0.
            x_dataS = self.scaler.fit_transform(x_data.T)
            x_feat = self.rff.toFeatures(x_dataS)
            w = None
            if self.weights is not None:
                w = self.weights.eval(x_data.T)

            self.model.fit(x_feat, y_data.T,
                           sample_weight=w,
                           epochs=nepoch,
                           validation_split=0.1,
                           callbacks=[lossHistory],
                           verbose=0,
                           batch_size=batch_size)

        end_time = datetime.now()
        print('')
        print("*********************************  End  *********************************")
        print()
        print("Training Finished, final loss is {}".format(lossHistory.losses[-1]))
        print('Duration: {}'.format(end_time - start_time))

        if plot:
            plt.plot(np.arange(len(lossHistory.losses)), lossHistory.losses)

        return self.model, lossHistory.losses

    # Prediction function
    def predict(self, x_test):
        start_time = datetime.now()
        x_testS = self.scaler.transform(x_test.T)
        x_feat = self.rff.toFeatures(x_testS)
        y_pred = self.model.predict(x_feat)
        # end_time = datetime.now()
        # print('\n')
        # print("*********************************  Prediction ends  *********************************")
        # print('\n')
        # print('Duration: {}'.format(end_time - start_time))

        return y_pred

    # Prediction function returning a delfi MoG
    def predict_mog_from_stats(self, ntest=1, alpha_pred=None, mu_pred=None, sigma_pred=None):
        alpha_pred = np.array(alpha_pred).reshape(-1, self.ncomp)
        mu_pred = np.array(mu_pred).reshape(-1, self.ncomp, self.outputd)
        sigma_pred = np.array(sigma_pred).reshape(-11, self.ncomp, self.outputd)
        mog = []
        for pt in range(ntest):
            a = alpha_pred[pt, :]
            ms = [mu_pred[pt, i, :] for i in range(self.ncomp)]
            Ss = []
            di = np.diag_indices(self.outputd)  # diagonal indices
            for i in range(self.ncomp):
                tmp = np.zeros((self.outputd, self.outputd))
                tmp[di] = sigma_pred[pt, i, :] ** 2
                Ss.append(tmp)
            mog.append(pdf.MoG(a=a, ms=ms, Ss=Ss))
        return mog[0]

    # Prediction function returning a delfi MoG
    def predict_mog(self, x_test):
        tfd = tfp.distributions
        start_time = datetime.now()
        x_testS = self.scaler.transform(x_test)
        x_feat = self.rff.toFeatures(x_testS)
        y_pred = self.model.predict(x_feat)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        # Builds the MoG
        # Parameters of the mixture
        ntest, dim = x_test.shape  # test dimensionality and number of queries
        comp = np.reshape(y_pred, [-1, self.outputd +
                                   int(0.5 * (self.outputd + 1) *
                                       self.outputd) + 1, self.ncomp])

        mu_pred = comp[:, :self.outputd, :]
        sigma_pred = comp[:, self.outputd:
                             self.outputd +
                             int(0.5 * (self.outputd + 1) *
                                 self.outputd), :]

        mu_pred = np.reshape(mu_pred, [-1, self.ncomp, self.outputd])
        sigma_pred = np.reshape(sigma_pred, [-1, self.ncomp,
                                             int(0.5 * (self.outputd + 1) *
                                                 self.outputd)])
        alpha_pred = comp[:, -1, :]

        mog = []
        for pt in range(ntest):
            a = alpha_pred[pt, :]
            ms = [mu_pred[pt, i, :] for i in range(self.ncomp)]
            Ss = []
            for comp_idx in range(self.ncomp):
                m = tfp.math.fill_triangular(sigma_pred[pt, comp_idx, :])
                tf.compat.v1.matrix_set_diag(m, tf.exp(tf.compat.v1.matrix_diag_part(m)))
                # sess = tf.compat.v1.Session()
                # with sess.as_default()
                #     L = m.eval()
                L = m.numpy()

                Ss.append(np.matmul(L, L.T))
            mog.append(pdf.MoG(a=a, ms=ms, Ss=Ss))
        return mog


class MDNN(object):
    def __init__(self, ncomp=10, nhidden=2, nunits=[24, 24], inputd=None, outputd=None,
                 weights=None, eps=0.):

        self.ncomp = ncomp  # number of mixture components
        self.eps = eps
        self.nhidden = nhidden  # number of hidden layers
        self.nunits = nunits  # number of units per hidden layer (integer or array)
        self.inputd = inputd  # dimensionality of the input
        self.outputd = outputd  # dimensionality of the output
        self.weights = weights  # sample weights

        def elu_modif(x, a=1.):
            e = 1e-15
            return ELU(alpha=a)(x) + 1. + e

        # Note: The output size will be (outputd + 2) * ncomp

        # For reference, not used at the moment
        def log_sum_exp(x, axis=None):
            """Log-sum-exp trick implementation"""
            x_max = K.max(x, axis=axis, keepdims=True)
            return K.log(K.sum(K.exp(x - x_max),
                               axis=axis, keepdims=True)) + x_max

        def tril_matrix(elements):
            tfd = tfp.distributions
            tril_m = tfp.math.fill_triangular(elements)
            tf.compat.v1.matrix_set_diag(tril_m, tf.add(tf.exp(tf.compat.v1.matrix_diag_part(tril_m)), self.eps))
            return tril_m

        def mean_log_Gaussian_like(y_true, parameters):
            # This version uses tensorflow_probability
            components = K.reshape(parameters, [-1, self.outputd +
                                                int(0.5 * (self.outputd + 1) *
                                                    self.outputd) + 1, self.ncomp])
            mu = components[:, :self.outputd, :]
            mu = K.reshape(mu, [-1, self.ncomp, self.outputd])
            sigma = components[:,
                    self.outputd: self.outputd + int(0.5 * (self.outputd + 1) * self.outputd), :]
            sigma = K.reshape(sigma, [-1, self.ncomp, int(0.5 * (self.outputd + 1) * self.outputd)])
            alpha = components[:, -1, :]

            # alpha = K.softmax(K.clip(alpha,1e-8,1.))
            alpha = K.clip(alpha, 1e-8, 1.)

            tfd = tfp.distributions
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=alpha),
                components_distribution=tfd.MultivariateNormalTriL(
                    loc=mu,
                    scale_tril=tril_matrix(sigma)))

            log_gauss = mix.log_prob(y_true)
            res = - K.mean(log_gauss)
            return res

        # This returns a tensor
        inputs = Input(shape=(self.inputd,))

        # Initializer with a particular seed
        initializer = lecun_normal(seed=1)

        # a layer instance is callable on a tensor, and returns a tensor
        nn = Dense(self.nunits[0], activation='tanh', kernel_initializer=initializer)(inputs)
        # nn = Dropout(0.05)(nn)

        for i in range(self.nhidden - 2):
            nn = Dense(self.nunits[i + 1], activation='tanh', kernel_initializer=initializer)(nn)
            # nn = Dropout(0.05)(nn)

        FC_mus = Dense(units=self.outputd * self.ncomp,
                       activation='linear',
                       kernel_initializer=initializer,
                       name='FC_mus')(nn)
        FC_sigmas_d = Dense(units=self.outputd * self.ncomp,
                            activation='linear',
                            kernel_initializer=initializer,
                            name='FC_sigmas_d')(nn)  # K.exp, W_regularizer=l2(1e-3)
        FC_sigmas = Dense(units=int(0.5 * (self.outputd - 1) * self.outputd * self.ncomp),
                          activation='linear',
                          kernel_initializer=initializer,
                          name='FC_sigmas')(nn)  # K.exp, W_regularizer=l2(1e-3)
        FC_alphas = Dense(units=self.ncomp,
                          activation='softmax',
                          kernel_initializer=initializer,
                          name='FC_alphas')(nn)

        output = concatenate([FC_mus, FC_sigmas_d, FC_sigmas, FC_alphas], axis=1)
        self.model = Model(inputs=inputs, outputs=output)

        # Note: Replace 'rmsprop' by 'adam' depending on your needs.
        self.model.compile('adam', loss=mean_log_Gaussian_like)

    # Training function
    def train(self, x_data, y_data, nepoch=1000, plot=False, save=False, batch_size=100, wandb=None):
        lossHistory = LossHistory()

        # checkpoint
        if save:

            print("\n\nTraining model....")
            filepath = "MDN--{epoch:02d}-{val_loss:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            start_time = datetime.now()
            if self.weights is not None:
                w = self.weights.eval(x_data.T)
            callbacks = [lossHistory, checkpoint]
            if wandb is not None:
                callbacks.append(WandbMetricsLogger())
            self.model.fit(x_data.T, y_data.T, epochs=nepoch,  # validation_split=0.1,
                           callbacks=callbacks, batch_size=batch_size,
                           sample_weight=w, verbose=1)
        else:
            if self.weights is not None:
                w = self.weights.eval(x_data.T)
            start_time = datetime.now()
            print("\nTraining model...")

            # print(self.model.layers[1].get_weights()[0][0][0])

            self.model.fit(x_data.T, y_data.T, epochs=nepoch,
                           batch_size=batch_size,
                           callbacks=[lossHistory],
                           verbose=0)

            # print(self.model.layers[1].get_weights()[0][0][0])

        # end_time = datetime.now()
        # print('')
        # print("*********************************  End  *********************************")
        # print()
        # print("Training Finished, final loss is {}".format(lossHistory.losses[-1]))
        # print('Duration: {}'.format(end_time - start_time))

        if plot:
            plt.plot(np.arange(len(lossHistory.losses)), lossHistory.losses)

        return self.model, lossHistory.losses

    # Prediction function
    def predict(self, x_test):
        start_time = datetime.now()
        y_pred = self.model.predict(x_test.T)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        return y_pred

    # Prediction function returning a delfi MoG
    def predict_mog(self, x_test):
        tfd = tfp.distributions
        start_time = datetime.now()
        y_pred = self.model.predict(x_test)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        # Builds the MoG
        # Parameters of the mixture
        ntest, dim = x_test.shape  # test dimensionality and number of queries
        comp = np.reshape(y_pred, [-1, self.outputd +
                                   int(0.5 * (self.outputd + 1) *
                                       self.outputd) + 1, self.ncomp])

        mu_pred = comp[:, :self.outputd, :]
        sigma_pred = comp[:, self.outputd:
                             self.outputd +
                             int(0.5 * (self.outputd + 1) *
                                 self.outputd), :]

        mu_pred = np.reshape(mu_pred, [-1, self.ncomp, self.outputd])
        sigma_pred = np.reshape(sigma_pred, [-1, self.ncomp,
                                             int(0.5 * (self.outputd + 1) *
                                                 self.outputd)])
        alpha_pred = comp[:, -1, :]

        mog = []
        for pt in range(ntest):
            a = alpha_pred[pt, :]
            ms = [mu_pred[pt, i, :] for i in range(self.ncomp)]
            Ss = []
            for comp_idx in range(self.ncomp):
                m = tfp.math.fill_triangular(sigma_pred[pt, comp_idx, :])
                tf.compat.v1.matrix_set_diag(m, tf.exp(tf.compat.v1.matrix_diag_part(m)))
                # sess = tf.compat.v1.Session()
                # with sess.as_default()
                #     L = m.eval()
                L = m.numpy()

                Ss.append(np.matmul(L, L.T))
            mog.append(pdf.MoG(a=a, ms=ms, Ss=Ss))
        return mog


class MDLSTM(object):
    def __init__(self, ncomp=10, nhidden=2, nunits=[50, 24, 24],
                 inputd=None, outputd=None, nsteps=200, weights=None, eps=0.):

        self.ncomp = ncomp  # number of mixture components
        self.eps = eps
        self.nhidden = nhidden  # number of hidden layers
        self.nunits = nunits  # number of units per hidden layer (integer or array)
        self.inputd = inputd  # dimensionality of the input
        self.nsteps = nsteps  # number of time steps in the
        self.outputd = outputd  # dimensionality of the output
        self.weights = weights  # sample weights

        def elu_modif(x, a=1.):
            e = 1e-15
            return ELU(alpha=a)(x) + 1. + e

        # Note: The output size will be (outputd + 2) * ncomp

        # For reference, not used at the moment
        def log_sum_exp(x, axis=None):
            """Log-sum-exp trick implementation"""
            x_max = K.max(x, axis=axis, keepdims=True)
            return K.log(K.sum(K.exp(x - x_max),
                               axis=axis, keepdims=True)) + x_max

        def tril_matrix(elements):
            tfd = tfp.distributions
            tril_m = tfp.math.fill_triangular(elements)
            tf.compat.v1.matrix_set_diag(tril_m, tf.add(tf.exp(tf.compat.v1.matrix_diag_part(tril_m)), self.eps))
            return tril_m

        def mean_log_Gaussian_like(y_true, parameters):
            # This version uses tensorflow_probability
            components = K.reshape(parameters, [-1, self.outputd +
                                                int(0.5 * (self.outputd + 1) *
                                                    self.outputd) + 1, self.ncomp])
            mu = components[:, :self.outputd, :]
            mu = K.reshape(mu, [-1, self.ncomp, self.outputd])
            sigma = components[:,
                    self.outputd: self.outputd + int(0.5 * (self.outputd + 1) * self.outputd), :]
            sigma = K.reshape(sigma, [-1, self.ncomp, int(0.5 * (self.outputd + 1) * self.outputd)])
            alpha = components[:, -1, :]

            # alpha = K.softmax(K.clip(alpha,1e-8,1.))
            alpha = K.clip(alpha, 1e-8, 1.)

            tfd = tfp.distributions
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=alpha),
                components_distribution=tfd.MultivariateNormalTriL(
                    loc=mu,
                    scale_tril=tril_matrix(sigma)))

            log_gauss = mix.log_prob(y_true)
            res = - K.mean(log_gauss)
            return res

        # This returns a tensor
        inputs = Input(shape=(self.nsteps, self.inputd))

        # Initializer with a particular seed
        initializer = lecun_normal(seed=1.)

        # Add the LSTM layer
        nn = LSTM(self.nunits[0])(inputs)

        # a layer instance is callable on a tensor, and returns a tensor
        # nn = Dense(self.nunits[0], activation='tanh', kernel_initializer=initializer)(nn)
        # nn = Dropout(0.05)(nn)

        for i in range(self.nhidden - 2):
            nn = Dense(self.nunits[i + 1], activation='tanh', kernel_initializer=initializer)(nn)
            # nn = Dropout(0.05)(nn)

        FC_mus = Dense(units=self.outputd * self.ncomp,
                       activation='linear',
                       kernel_initializer=initializer,
                       name='FC_mus')(nn)
        FC_sigmas_d = Dense(units=self.outputd * self.ncomp,
                            activation='linear',
                            kernel_initializer=initializer,
                            name='FC_sigmas_d')(nn)  # K.exp, W_regularizer=l2(1e-3)
        FC_sigmas = Dense(units=int(0.5 * (self.outputd - 1) * self.outputd * self.ncomp),
                          activation='linear',
                          kernel_initializer=initializer,
                          name='FC_sigmas')(nn)  # K.exp, W_regularizer=l2(1e-3)
        FC_alphas = Dense(units=self.ncomp,
                          activation='softmax',
                          kernel_initializer=initializer,
                          name='FC_alphas')(nn)

        output = concatenate([FC_mus, FC_sigmas_d, FC_sigmas, FC_alphas], axis=1)
        self.model = Model(inputs=inputs, outputs=output)

        # Note: Replace 'rmsprop' by 'adam' depending on your needs.
        self.model.compile('adam', loss=mean_log_Gaussian_like)

    # Training function
    def train(self, x_data, y_data, nepoch=1000, plot=False, save=False, batch_size=100, wandb=None):
        lossHistory = LossHistory()

        # checkpoint
        if save:
            print("Training model....")
            filepath = "MDN--{epoch:02d}-{val_loss:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            start_time = datetime.now()
            w = None
            if self.weights is not None:
                w = self.weights.eval(x_data.T)
            callbacks = [lossHistory, checkpoint]
            if wandb is not None:
                callbacks.append(WandbMetricsLogger())
            self.model.fit(x_data.T, y_data.T, epochs=nepoch,  # validation_split=0.1,
                           callbacks=callbacks, batch_size=batch_size,
                           sample_weight=w, verbose=0)
        else:
            w = None
            if self.weights is not None:
                w = self.weights.eval(x_data.T)
            start_time = datetime.now()
            print("Training model....")
            self.model.fit(x_data.T, y_data.T, epochs=nepoch,
                           batch_size=batch_size,
                           callbacks=[lossHistory],
                           sample_weight=w,
                           verbose=1)

        end_time = datetime.now()
        print('')
        print("*********************************  End  *********************************")
        print()
        print("Training Finished, final loss is {}".format(lossHistory.losses[-1]))
        print('Duration: {}'.format(end_time - start_time))

        if plot:
            plt.plot(np.arange(len(lossHistory.losses)), lossHistory.losses)

        return self.model, lossHistory.losses

    # Prediction function
    def predict(self, x_test):
        start_time = datetime.now()
        y_pred = self.model.predict(x_test.T)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        return y_pred

    # Prediction function returning a delfi MoG
    def predict_mog(self, x_test):
        tfd = tfp.distributions
        x_test = x_test.T
        ntest = x_test.shape[1]
        start_time = datetime.now()
        # Prepares the data for LSTMs

        x_test_fixed = np.zeros([self.nsteps, self.inputd, ntest])
        for i in range(ntest):
            x_test_fixed[:, :, i] = x_test[:, i].reshape([self.nsteps, self.inputd])
        x_test_fixed = np.swapaxes(x_test_fixed, 0, 1)
        x_test_fixed = np.swapaxes(x_test_fixed, 0, 2)
        y_pred = self.model.predict(x_test_fixed)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        # Builds the MoG
        # Parameters of the mixture
        # ntest, dim = x_test.shape  # test dimensionality and number of queries
        comp = np.reshape(y_pred, [-1, self.outputd +
                                   int(0.5 * (self.outputd + 1) *
                                       self.outputd) + 1, self.ncomp])

        mu_pred = comp[:, :self.outputd, :]
        sigma_pred = comp[:, self.outputd:
                             self.outputd +
                             int(0.5 * (self.outputd + 1) *
                                 self.outputd), :]

        mu_pred = np.reshape(mu_pred, [-1, self.ncomp, self.outputd])
        sigma_pred = np.reshape(sigma_pred, [-1, self.ncomp,
                                             int(0.5 * (self.outputd + 1) *
                                                 self.outputd)])
        alpha_pred = comp[:, -1, :]

        mog = []
        for pt in range(ntest):
            a = alpha_pred[pt, :]
            ms = [mu_pred[pt, i, :] for i in range(self.ncomp)]
            Ss = []
            for comp_idx in range(self.ncomp):
                m = tfp.math.fill_triangular(sigma_pred[pt, comp_idx, :])
                tf.compat.v1.matrix_set_diag(m, tf.exp(tf.compat.v1.matrix_diag_part(m)))
                # sess = tf.compat.v1.Session()
                # with sess.as_default()
                #     L = m.eval()
                L = m.numpy()

                Ss.append(np.matmul(L, L.T))
            mog.append(pdf.MoG(a=a, ms=ms, Ss=Ss))
        return mog


class MDRFFLSTM(object):
    """
    Mixture density model with LSTM and RFFs instead of a neural network. The output of the LSTM is passed to
    RFFs which are then used to compute the parameters of the mixture.
    """
    def __init__(self, ncomp=10, nfeat=154, nunits=50, kernel="RBF", sigma=1.,
                 quasiRandom=True, cosOnly=False, inputd=None, outputd=None, nsteps=200, weights=None, eps=0.):

        self.ncomp = ncomp  # number of mixture components
        self.eps = eps
        self.nunits = nunits  # number of output units for LSTM
        self.inputd = inputd  # dimensionality of the input
        self.nsteps = nsteps  # number of time steps in the
        self.outputd = outputd  # dimensionality of the output
        self.weights = weights  # sample weights
        self.nfeat = nfeat  # number of RFFs
        self.quasiRandom = quasiRandom
        self.cosOnly = cosOnly
        self.sigma = sigma * np.ones(self.nunits)
        self.kernel = kernel
        self.rff_tf = RFF_TF(self.nfeat, self.nunits, self.sigma,
                             self.cosOnly, self.quasiRandom, self.kernel)

        def elu_modif(x, a=1.):
            e = 1e-15
            return ELU(alpha=a)(x) + 1. + e

        # Note: The output size will be (outputd + 2) * ncomp

        # For reference, not used at the moment
        def log_sum_exp(x, axis=None):
            """Log-sum-exp trick implementation"""
            x_max = K.max(x, axis=axis, keepdims=True)
            return K.log(K.sum(K.exp(x - x_max),
                               axis=axis, keepdims=True)) + x_max

        def tril_matrix(elements):
            tfd = tfp.distributions
            tril_m = tfp.math.fill_triangular(elements)
            tf.compat.v1.matrix_set_diag(tril_m, tf.add(tf.exp(tf.compat.v1.matrix_diag_part(tril_m)), self.eps))
            return tril_m

        def mean_log_Gaussian_like(y_true, parameters):
            # This version uses tensorflow_probability
            components = K.reshape(parameters, [-1, self.outputd +
                                                int(0.5 * (self.outputd + 1) *
                                                    self.outputd) + 1, self.ncomp])
            mu = components[:, :self.outputd, :]
            mu = K.reshape(mu, [-1, self.ncomp, self.outputd])
            sigma = components[:,
                    self.outputd: self.outputd + int(0.5 * (self.outputd + 1) * self.outputd), :]
            sigma = K.reshape(sigma, [-1, self.ncomp, int(0.5 * (self.outputd + 1) * self.outputd)])
            alpha = components[:, -1, :]

            # alpha = K.softmax(K.clip(alpha,1e-8,1.))
            alpha = K.clip(alpha, 1e-8, 1.)

            tfd = tfp.distributions
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=alpha),
                components_distribution=tfd.MultivariateNormalTriL(
                    loc=mu,
                    scale_tril=tril_matrix(sigma)))

            log_gauss = mix.log_prob(y_true)
            res = - K.mean(log_gauss)
            return res

        # This returns a tensor
        inputs = Input(shape=(None, self.inputd))

        # Initializer with a particular seed
        initializer = lecun_normal(seed=1.)

        # Add the LSTM layer
        nn = LSTM(self.nunits)(inputs)

        # Computes the random Fourier features
        rff = Lambda(self.rff_tf.toFeatures)(nn)

        FC_mus = Dense(units=self.outputd * self.ncomp,
                       activation='linear',
                       kernel_initializer=initializer,
                       name='FC_mus')(rff)
        FC_sigmas_d = Dense(units=self.outputd * self.ncomp,
                            activation='linear',
                            kernel_initializer=initializer,
                            name='FC_sigmas_d')(rff)  # K.exp, W_regularizer=l2(1e-3)
        FC_sigmas = Dense(units=int(0.5 * (self.outputd - 1) * self.outputd * self.ncomp),
                          activation='linear',
                          kernel_initializer=initializer,
                          name='FC_sigmas')(rff)  # K.exp, W_regularizer=l2(1e-3)
        FC_alphas = Dense(units=self.ncomp,
                          activation='softmax',
                          kernel_initializer=initializer,
                          name='FC_alphas')(rff)

        output = concatenate([FC_mus, FC_sigmas_d, FC_sigmas, FC_alphas], axis=1)
        self.model = Model(inputs=inputs, outputs=output)

        # Note: Replace 'rmsprop' by 'adam' depending on your needs.
        self.model.compile('adam', loss=mean_log_Gaussian_like)

    # Training function
    def train(self, x_data, y_data, nepoch=1000, plot=False, save=False, batch_size=100, wandb=None):
        lossHistory = LossHistory()

        # checkpoint
        if save:
            print("Training model....")
            filepath = "MDN--{epoch:02d}-{val_loss:.2f}.hdf5"
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            start_time = datetime.now()
            w = None
            if self.weights is not None:
                w = self.weights.eval(x_data.T)
            callbacks = [lossHistory, checkpoint]
            if wandb is not None:
                callbacks.append(WandbMetricsLogger())
            self.model.fit(x_data.T, y_data.T, epochs=nepoch,  # validation_split=0.1,
                           callbacks=callbacks, batch_size=batch_size,
                           sample_weight=w, verbose=0)
        else:
            w = None
            if self.weights is not None:
                w = self.weights.eval(x_data.T)
            start_time = datetime.now()
            print("Training model....")
            self.model.fit(x_data.T, y_data.T, epochs=nepoch,
                           batch_size=batch_size,
                           callbacks=[lossHistory],
                           sample_weight=w,
                           verbose=1)

        end_time = datetime.now()
        print('')
        print("*********************************  End  *********************************")
        print()
        print("Training Finished, final loss is {}".format(lossHistory.losses[-1]))
        print('Duration: {}'.format(end_time - start_time))

        if plot:
            plt.plot(np.arange(len(lossHistory.losses)), lossHistory.losses)

        return self.model, lossHistory.losses

    # Prediction function
    def predict(self, x_test):
        start_time = datetime.now()
        y_pred = self.model.predict(x_test.T)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        return y_pred

    # Prediction function returning a delfi MoG
    def predict_mog(self, x_test):
        tfd = tfp.distributions
        x_test = x_test.T
        ntest = x_test.shape[1]
        start_time = datetime.now()
        # Prepares the data for LSTMs

        x_test_fixed = np.zeros([self.nsteps, self.inputd, ntest])
        for i in range(ntest):
            x_test_fixed[:, :, i] = x_test[:, i].reshape([self.nsteps, self.inputd])
        x_test_fixed = np.swapaxes(x_test_fixed, 0, 1)
        x_test_fixed = np.swapaxes(x_test_fixed, 0, 2)
        y_pred = self.model.predict(x_test_fixed)
        end_time = datetime.now()
        print('\n')
        print("*********************************  Prediction ends  *********************************")
        print('\n')
        print('Duration: {}'.format(end_time - start_time))

        # Builds the MoG
        # Parameters of the mixture
        # ntest, dim = x_test.shape  # test dimensionality and number of queries
        comp = np.reshape(y_pred, [-1, self.outputd +
                                   int(0.5 * (self.outputd + 1) *
                                       self.outputd) + 1, self.ncomp])

        mu_pred = comp[:, :self.outputd, :]
        sigma_pred = comp[:, self.outputd:
                             self.outputd +
                             int(0.5 * (self.outputd + 1) *
                                 self.outputd), :]

        mu_pred = np.reshape(mu_pred, [-1, self.ncomp, self.outputd])
        sigma_pred = np.reshape(sigma_pred, [-1, self.ncomp,
                                             int(0.5 * (self.outputd + 1) *
                                                 self.outputd)])
        alpha_pred = comp[:, -1, :]

        mog = []
        for pt in range(ntest):
            a = alpha_pred[pt, :]
            ms = [mu_pred[pt, i, :] for i in range(self.ncomp)]
            Ss = []
            for comp_idx in range(self.ncomp):
                m = tfp.math.fill_triangular(sigma_pred[pt, comp_idx, :])
                tf.compat.v1.matrix_set_diag(m, tf.exp(tf.compat.v1.matrix_diag_part(m)))
                # sess = tf.compat.v1.Session()
                # with sess.as_default()
                #     L = m.eval()
                L = m.numpy()

                Ss.append(np.matmul(L, L.T))
            mog.append(pdf.MoG(a=a, ms=ms, Ss=Ss))
        return mog

