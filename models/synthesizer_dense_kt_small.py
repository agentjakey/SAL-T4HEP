import os
import time
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf 
from tensorflow.keras import layers, Input, Model 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# ---------------------------
# Parse args for mode + variant
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "test"], default="train", help="Run training or testing")
args = parser.parse_args()
model_name = 'dense_kt_small'

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
        #alpha and beta are trainable scalars that modulate shape of tanh
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
        v = self.split_heads(v, batch_size)  

        # Generate synthetic attention scores
        dense_scores = self.attn_generator(x)  
        dense_scores = tf.expand_dims(dense_scores, 1)  
        attn_weights = tf.nn.softmax(dense_scores, axis=-1)  

        # Compute attention output
        attn_output = tf.matmul(attn_weights, v)  
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  
        concat = tf.reshape(attn_output, (batch_size, -1, self.d_model))  
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
# Memory growth setting
# ---------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ---------------------------
# GPU peak memory + FLOPs
# ---------------------------
def profile_gpu_memory_during_inference(
    model: tf.keras.Model,
    input_data: np.ndarray,
) -> tuple[float, float]:
    """
    Runs one forward pass in @tf.function and returns
    (current_gpu_mb, peak_gpu_mb) allocated during that call.
    """
    # reset stats so we get a fresh peak measurement
    tf.config.experimental.reset_memory_stats("GPU:0")

    @tf.function
    def infer(x):
        return model(x, training=False)

    # warm-up to allocate buffers
    _ = infer(input_data[:1])
    # actual profiling
    _ = infer(input_data)

    mem = tf.config.experimental.get_memory_info("GPU:0")
    current_mb = mem["current"] / (1024**2)
    peak_mb = mem["peak"] / (1024**2)
    return current_mb, peak_mb


def get_flops(model, input_shape):
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph,
    )

    inp = tf.TensorSpec(input_shape, tf.float32)
    func = tf.function(model).get_concrete_function(inp)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(func)
    with tf.Graph().as_default() as g:
        tf.compat.v1.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=g, run_meta=run_meta, cmd="op", options=opts
        )
        return flops.total_float_ops

# ---------------------------
# Build Synthesizer Transformer Classifier
# ---------------------------
def build_synthesizer_transformer_classifier(
    num_particles, feature_dim,
    d_model=8, d_ff=8, output_dim=5,
    num_heads=2): 

    inputs = layers.Input((num_particles, feature_dim)) 
    x = layers.Dense(d_model, activation='relu')(inputs) 
    x = SynthesizerTransformerBlock(
        d_model = d_model, d_ff = d_ff, output_dim = output_dim, num_heads = num_heads
    )(x) 
    x = AggregationLayer('max')(x) 
    x = layers.Dense(d_model, activation='relu')(x) 

    activation = 'sigmoid' if output_dim == 1 else 'softmax'  
    outputs = layers.Dense(output_dim, activation=activation)(x) 
    return Model(inputs, outputs) 

# ---------------------------
# Sorting helper
# ---------------------------
def apply_sorting(x, sort_by):
    if sort_by == "pt":
        key = x[:, :, 0]
    elif sort_by == "eta":
        key = x[:, :, 1]
    elif sort_by == "phi":
        key = x[:, :, 2]
    elif sort_by == "delta_R":
        key = np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
    elif sort_by == "kt":
        key = x[:, :, 0] * np.sqrt(x[:, :, 1] ** 2 + x[:, :, 2] ** 2)
    else:
        return x
    idx = np.argsort(key, axis=1)[:, ::-1]
    return np.take_along_axis(x, idx[:, :, None], axis=1)

# Load data
X_train = np.load("/j-jepa-vol/l1-jet-id/data/jetid/processed/x_train_robust_150const_ptetaphi_sorted_kt.npy")
y_train = np.load("/j-jepa-vol/l1-jet-id/data/jetid/processed/y_train_robust_150const_ptetaphi.npy")
X_test = np.load("/j-jepa-vol/l1-jet-id/data/jetid/processed/x_val_robust_150const_ptetaphi.npy")
X_test = apply_sorting(X_test, sort_by="kt")
y_test = np.load("/j-jepa-vol/l1-jet-id/data/jetid/processed/y_val_robust_150const_ptetaphi.npy")


# Convert labels to categorical
num_classes = len(np.unique(y_train))
y_train_cat = y_train
y_test_cat = y_test


# Build and compile the model
model = build_synthesizer_transformer_classifier(
    num_particles=X_train.shape[1],
    feature_dim=X_train.shape[2],
    output_dim=y_train_cat.shape[1]
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ---------------------------
# Total parameters
# ---------------------------
print(f"Total parameters: {model.count_params():,}")

# ---------------------------
# Callbacks & Logging
# ---------------------------
save_dir = f"/j-jepa-vol/l1-jet-id/synthesizer_training/{model_name}"
os.makedirs(save_dir, exist_ok=True)
best_weights_path = os.path.join(save_dir, f"{model_name}_best.weights.h5")

# ---------------------------
# TRAIN MODE
# ---------------------------
if args.mode == "train":
    ckpt = ModelCheckpoint(best_weights_path, save_best_only=True, monitor="val_loss", verbose=1, save_weights_only=True)
    early = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1)

    schedule = [(128, 200), (256, 200), (512, 200), (1024, 200), (2048, 600)]
    ce = 0
    histories = []

    start_time = time.time()
    for bs, ep in schedule:
        tf.keras.backend.set_value(model.optimizer.lr, 1e-3)
        hist = model.fit(
            X_train, y_train_cat,
            validation_split=0.2,
            initial_epoch=ce,
            epochs=ce + ep,
            batch_size=bs,
            callbacks=[ckpt, early],
            verbose=1
        )
        histories.append(hist)
        ce += ep
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} sec")

    # Save curves
    train_loss = np.concatenate([h.history["loss"] for h in histories])
    val_loss = np.concatenate([h.history["val_loss"] for h in histories])
    train_acc = np.concatenate([h.history["accuracy"] for h in histories])
    val_acc = np.concatenate([h.history["val_accuracy"] for h in histories])

    np.save(os.path.join(save_dir, f"{model_name}_train_loss.npy"), train_loss)
    np.save(os.path.join(save_dir, f"{model_name}_val_loss.npy"), val_loss)
    np.save(os.path.join(save_dir, f"{model_name}_train_accuracy.npy"), train_acc)
    np.save(os.path.join(save_dir, f"{model_name}_val_accuracy.npy"), val_acc)

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_accuracy_curve.png"))
    plt.close()

# ---------------------------
# TEST MODE
# ---------------------------
if args.mode == "test":
    if not os.path.exists(best_weights_path):
        raise FileNotFoundError(f"No best weights found for {model_name}")
    model.load_weights(best_weights_path)
    # FLOPs one forward pass 
    input_shape_for_flops = (1,) + tuple(X_test.shape[1:])
    try:
        flops = int(get_flops(model, input_shape_for_flops))
        print(f"Estimated FLOPs (batch=1): {flops:,}")
    except Exception as e:
        print(f"[warn] FLOPs estimation failed: {e}")

    # Peak GPU memory
    try:
        if len(tf.config.experimental.list_physical_devices("GPU")) > 0:
            x1 = X_test[:1].astype("float32", copy=False)  
            @tf.function                              
            def _infer(z): return model(z, training=False)
            _ = _infer(x1)                            
            current_mb, peak_mb = profile_gpu_memory_during_inference(model, x1)  
            print(f"GPU memory (one forward pass) — current: {current_mb:.2f} MB | peak: {peak_mb:.2f} MB")
        else:
            print("No GPU detected; skipping GPU memory profiling.")
    except Exception as e:
        print(f"[warn] GPU memory profiling failed: {e}")

    preds = model.predict(X_test)
    pred_classes = np.argmax(preds, axis=1)
    roc = roc_auc_score(y_test_cat, preds, multi_class="ovr")
    acc = accuracy_score(np.argmax(y_test_cat, axis=1), pred_classes)
    print(f"Variant: {model_name}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"ROC AUC Score: {roc:.4f}")
