############### IMPORT PACKAGES ##################

import numpy as np
from numpy.linalg import inv as inv  # Used in kalman filter

# Used for naive bayes decoder
try:
    import statsmodels.api as sm
except ImportError:
    print(
        "\nWARNING: statsmodels is not installed. You will be unable to use the Naive Bayes Decoder"
    )
    pass
try:
    import math
except ImportError:
    print(
        "\nWARNING: math is not installed. You will be unable to use the Naive Bayes Decoder"
    )
    pass
try:
    from scipy.spatial.distance import pdist
    from scipy.spatial.distance import squareform
    from scipy.stats import norm
    from scipy.spatial.distance import cdist
except ImportError:
    print(
        "\nWARNING: scipy is not installed. You will be unable to use the Naive Bayes Decoder"
    )
    pass


# Import scikit-learn (sklearn) if it is installed
try:
    from sklearn import linear_model  # For Wiener Filter and Wiener Cascade
    from sklearn.svm import SVR  # For support vector regression (SVR)
    from sklearn.svm import SVC  # For support vector classification (SVM)
except ImportError:
    print(
        "\nWARNING: scikit-learn is not installed. You will be unable to use the Wiener Filter or Wiener Cascade Decoders"
    )
    pass

# Import functions for Keras if Keras is installed
# Note that Keras has many more built-in functions that I have not imported because I have not used them
# But if you want to modify the decoders with other functions (e.g. regularization), import them here
try:
    import keras
    keras_v1 = int(keras.__version__[0]) <= 1
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, SimpleRNN, GRU, Activation, Dropout
    from keras.utils import np_utils
except ImportError:
    print(
        "\nWARNING: Keras package is not installed. You will be unable to use all neural net decoders"
    )
    pass

try:
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    print(
        "\nWARNING: Sklearn OneHotEncoder not installed. You will be unable to use XGBoost for Classification"
    )
    pass


class LSTMRegression(object):
    """
    Class for the long short term regression (GRU) decoder

    Parameters
    ----------
    units: integer, optional, default 400
        Number of hidden units in each layer
        
    units: Number of hidden units in each layer

    dropout: decimal, optional, default 0
        Proportion of units that get dropped out

    num_epochs: integer, optional, default 10
        Number of epochs used for training

    verbose: binary, optional, default=0
        Whether to show progress of the fit after each epoch
    """

    def __init__(self, units=400, dropout=0, num_epochs=10, verbose=0):
        self.units = units
        self.dropout = dropout
        self.num_epochs = num_epochs
        self.verbose = verbose

    def fit(self, X_train, y_train):
        """
        Train LSTM Decoder

        Parameters
        ----------
        X_train: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """

        model = Sequential()  # Declare model
        # Add recurrent layer
        if keras_v1:
            model.add(
                LSTM(
                    self.units,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    dropout_W=self.dropout,
                    dropout_U=self.dropout,
                )
            )  # Within recurrent layer, include dropout
        else:
            model.add(
                LSTM(
                    self.units,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    dropout=self.dropout,
                    recurrent_dropout=self.dropout,
                )
            )  # Within recurrent layer, include dropout
        if self.dropout != 0:
            model.add(
                Dropout(self.dropout)
            )  # Dropout some units (recurrent layer output units)

        # Add dense connections to output layer
        model.add(Dense(y_train.shape[1]))

        # Fit model (and set fitting parameters)
        model.compile(
            loss="mse", optimizer="rmsprop", metrics=["accuracy"]
        )  # Set loss function and optimizer
        if keras_v1:
            model.fit(
                X_train, y_train, nb_epoch=self.num_epochs, verbose=self.verbose
            )  # Fit the model
        else:
            model.fit(
                X_train, y_train, epochs=self.num_epochs, verbose=self.verbose
            )  # Fit the model
        self.model = model

    def predict(self, X_test):
        """
        Predict outcomes using trained LSTM Decoder

        Parameters
        ----------
        X_test: numpy 3d array of shape [n_samples,n_time_bins,n_neurons]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_test)  # Make predictions
        return y_test_predicted
