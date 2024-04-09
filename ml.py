import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Assuming your data is loaded into a DataFrame `df`
data = {
    'Node 0': [1, 1, 1, 1],
    'Node 1': [1, 1, 0, 1],
    'Node 2': [1, 1, 1, 1],
    'Node 3': [1, 0, 1, 0],
    'Node 4': [1, 1, 1, 1],
    'Node 5': [1, 1, 1, 1],
    'Node 6': [1, 0, 1, 1],
    'Node 7': [1, 1, 1, 1],
    'Node 8': [1, 1, 0, 1],
    'Node 9': [1, 1, 1, 1]
}
df = pd.DataFrame(data)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Define the autoencoder network architecture
input_dim = scaled_data.shape[1]
encoding_dim = 4

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
encoder = Dense(encoding_dim // 2, activation="relu")(encoder)

decoder = Dense(encoding_dim // 2, activation="relu")(encoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

autoencoder.fit(scaled_data, scaled_data,
                epochs=100,
                batch_size=2,
                shuffle=True,
                validation_split=0.2,
                verbose=1)

reconstructed_data = autoencoder.predict(scaled_data)

# Compute reconstruction error
reconstruction_error = np.mean(np.abs(scaled_data - reconstructed_data), axis=1)

# Examine the distribution of reconstruction error and choose a threshold
error_df = pd.DataFrame({'reconstruction_error': reconstruction_error})
error_df.describe()

# Set a threshold based on your observation of the error distribution
threshold = error_df['reconstruction_error'].quantile(0.9)  # Adjust based on the error distribution

anomalies = reconstruction_error > threshold

print("Anomalies found at indices:", np.where(anomalies)[0])
