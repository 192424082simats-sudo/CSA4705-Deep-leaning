import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

encoder_input = Input(shape=(5, 1))
encoder = LSTM(16, return_state=True)
_, h, c = encoder(encoder_input)

decoder_input = Input(shape=(5, 1))
decoder = LSTM(16, return_sequences=True)
decoder_output = decoder(decoder_input, initial_state=[h, c])

output = Dense(1)(decoder_output)

model = Model([encoder_input, decoder_input], output)
model.compile(optimizer='adam', loss='mse')

X = np.random.rand(100, 5, 1)
Y = np.random.rand(100, 5, 1)

model.fit([X, X], Y, epochs=2, verbose=0)

prediction = model.predict([X[:1], X[:1]], verbose=0)

print("Model training completed")
print("Input shape:", X.shape)
print("Output shape:", prediction.shape)
print("Prediction:", prediction)
