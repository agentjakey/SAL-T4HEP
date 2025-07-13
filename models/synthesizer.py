import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------------------------
# Aggregation Layer
# ---------------------------
class AggregationLayer(layers.Layer):
    def __init__(self, aggreg="mean", **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)
        self.aggreg = aggreg

    def call(self, inputs, training=None):
        if self.aggreg == "mean":
            return tf.reduce_mean(inputs, axis=1)
        elif self.aggreg == "max":
            return tf.reduce_max(inputs, axis=1)
        else:
            raise ValueError("Unsupported aggregation: use 'mean' or 'max'.")

# ---------------------------
# Dynamic Tanh Activation
# ---------------------------
class DynamicTanh(layers.Layer):
    def __init__(self, **kwargs):
        super(DynamicTanh, self).__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name="alpha", shape=(1,), initializer='ones', trainable=True)
        self.beta = self.add_weight(name="beta", shape=(1,), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, inputs, training=None):
        return tf.math.tanh(self.alpha * inputs + self.beta)

# ---------------------------
# Dense Synthesizer Attention Layer
# ---------------------------
class DenseSynthesizerAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

    def build(self, input_shape):
        self.seq_len = input_shape[1]

        self.value_proj = self.add_weight(
            name="value_proj",
            shape=(self.d_model, self.d_model),
            initializer="glorot_uniform",
            trainable=True
        )

        self.attn_generator = tf.keras.Sequential([
            layers.Dense(self.seq_len, activation='relu'),
            layers.Dense(self.seq_len)
        ])

        self.out_proj = layers.Dense(self.d_model)
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]

        # Value projection
        v = tf.matmul(x, self.value_proj)
        v = self.split_heads(v, batch_size)  # shape: [B, h, L, d]

        # Generate synthetic attention scores
        dense_scores = self.attn_generator(x)  # [B, L, L]
        dense_scores = tf.expand_dims(dense_scores, 1)  # [B, 1, L, L]
        attn_weights = tf.nn.softmax(dense_scores, axis=-1)  # [B, 1, L, L]

        # Compute attention output
        attn_output = tf.matmul(attn_weights, v)  # [B, h, L, d]
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  # [B, L, h, d]
        concat = tf.reshape(attn_output, (batch_size, -1, self.d_model))  # [B, L, D]
        return self.out_proj(concat)

# ---------------------------
# Synthesizer Transformer Block
# ---------------------------
class SynthesizerTransformerBlock(layers.Layer):
    def __init__(self, d_model, d_ff, output_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.attn = DenseSynthesizerAttention(d_model, num_heads)
        self.act1 = DynamicTanh()
        self.act2 = DynamicTanh()
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_ff, activation='relu'),
            layers.Dense(d_model)
        ])

    def call(self, x, training=None):
        attn_out = self.attn(x, training=training)
        out1 = self.act1(x + attn_out, training=training)
        ffn_out = self.ffn(out1, training=training)
        return self.act2(out1 + ffn_out, training=training)

# ---------------------------
# Build Synthesizer Transformer Classifier
# ---------------------------
def build_synthesizer_transformer_classifier(
    num_particles, feature_dim,
    d_model=16, d_ff=16, output_dim=16,
    num_heads=4):

    inputs = layers.Input((num_particles, feature_dim))
    x = layers.Dense(d_model, activation='relu')(inputs)
    x = SynthesizerTransformerBlock(
        d_model, d_ff, output_dim, num_heads
    )(x)
    x = AggregationLayer('max')(x)
    x = layers.Dense(d_model, activation='relu')(x)

    activation = 'sigmoid' if output_dim == 1 else 'softmax'
    outputs = layers.Dense(output_dim, activation=activation)(x)
    return Model(inputs, outputs)
