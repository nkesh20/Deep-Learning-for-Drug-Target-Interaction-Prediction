import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras.models import Model


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim), ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def build_transformer_model(config):
    # Inputs
    drug_input = Input(shape=(config['max_smi_len'],))
    protein_input = Input(shape=(config['max_seq_len'],))

    # Drug embedding
    drug_embedding = tf.keras.layers.Embedding(input_dim=config['charsmiset_size'] + 1,
                                               output_dim=128,
                                               input_length=config['max_smi_len'])(drug_input)

    # Protein embedding
    protein_embedding = tf.keras.layers.Embedding(input_dim=config['charseqset_size'] + 1,
                                                  output_dim=128,
                                                  input_length=config['max_seq_len'])(protein_input)

    # Transformer layers
    for _ in range(config['num_layers']):
        drug_transformer = TransformerBlock(128, config['num_heads'], config['ff_dim'], config['dropout_rate'])(
            drug_embedding)
        protein_transformer = TransformerBlock(128, config['num_heads'], config['ff_dim'], config['dropout_rate'])(
            protein_embedding)
        drug_embedding = drug_transformer
        protein_embedding = protein_transformer

    # Global average pooling
    drug_pooled = tf.keras.layers.GlobalAveragePooling1D()(drug_embedding)
    protein_pooled = tf.keras.layers.GlobalAveragePooling1D()(protein_embedding)

    # Concatenate
    concat = Concatenate()([drug_pooled, protein_pooled])

    # Fully connected layers
    fc1 = Dense(1024, activation='relu')(concat)
    fc2 = Dense(512, activation='relu')(fc1)
    output = Dense(1, activation='linear')(fc2)

    model = Model(inputs=[drug_input, protein_input], outputs=[output])
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model