# nn_lstm.py
# Template for LSTM forecasting/classification with Keras.
# This is a template and may require: pip install tensorflow
try:
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    # Template-only: provide instruction string for users
    pass

lstm_template = """
# pip install tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm(input_timesteps, input_features, units=64):
    model = Sequential([
        LSTM(units, input_shape=(input_timesteps, input_features)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# X shape: (n_samples, timesteps, features); y shape: (n_samples,)
# model = build_lstm(input_timesteps=24, input_features=3)
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
#                     epochs=50, batch_size=32, callbacks=[EarlyStopping(patience=5)])
# y_pred = model.predict(X_test).ravel()
"""
