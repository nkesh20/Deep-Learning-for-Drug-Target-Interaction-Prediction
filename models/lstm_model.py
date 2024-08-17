import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model

def build_lstm_model(config):
    # Inputs
    drug_input = Input(shape=(config['max_smi_len'],))
    protein_input = Input(shape=(config['max_seq_len'],))

    # Drug embedding
    drug_embedding = tf.keras.layers.Embedding(input_dim=config['charsmiset_size'] + 1,
                                               output_dim=128,
                                               input_length=config['max_smi_len'])(drug_input)
    drug_lstm = LSTM(config['lstm_units'])(drug_embedding)

    # Protein embedding
    protein_embedding = tf.keras.layers.Embedding(input_dim=config['charseqset_size'] + 1,
                                                  output_dim=128,
                                                  input_length=config['max_seq_len'])(protein_input)
    protein_lstm = LSTM(config['lstm_units'])(protein_embedding)

    # Concatenate
    concat = Concatenate()([drug_lstm, protein_lstm])

    # Fully connected layers
    fc1 = Dense(1024, activation='relu')(concat)
    fc2 = Dense(512, activation='relu')(fc1)
    output = Dense(1, activation='linear')(fc2)

    model = Model(inputs=[drug_input, protein_input], outputs=[output])
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model