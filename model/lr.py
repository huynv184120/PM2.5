from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

def linear_model_construction(input_dim, output_dim, optimizer, log_dir, is_training=True):
    model = Sequential()
    model.add(Dense(output_dim, input_shape=(None,input_dim)))    

    if is_training:
        return model
    else:
        print("Load model from: {}".format(log_dir))
        model.load_weights(log_dir + 'best_model.hdf5')
        model.compile(optimizer=optimizer, loss='mse')
        return model