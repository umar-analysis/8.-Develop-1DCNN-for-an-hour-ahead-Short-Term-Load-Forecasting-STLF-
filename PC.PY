from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.regularizers import l2

def build(time_steps, num_features, reg=0.0005):
    model = Sequential()
    model.add(LSTM(64, input_shape=(time_steps, num_features), kernel_regularizer=l2(reg)))
    model.add(Dense(1))
    return model
