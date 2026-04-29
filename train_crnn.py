import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LSTM, Dense, Reshape, Concatenate, BatchNormalization, ELU, Dropout
from tensorflow.keras.models import Model
import scipy.signal
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend to avoid Tkinter thread loop errors
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
FS = 12800
NPERSEG = 128
NOVERLAP =64
EPOCHS = 20
BATCH_SIZE = 64
script_dir = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(script_dir, "dataset_output")
TOTAL_SCENARIOS = 1000000

# We use global mean/std estimated from the old smaller run or approximated cleanly:
# STFT log1p stats approximations:
# (Allows BN to handle the rest efficiently without loading 10GB globally)
APPROX_MEAN = 0.0 # Real/imag pairs roughly center around 0
APPROX_STD = 1.0

# -------------------------------------------------------------
# Data Processing via tf.data
# -------------------------------------------------------------
def preprocess_tf_map(x_raw, y_raw):
    """
    TensorFlow native graph mapping to convert raw sequences -> STFT -> Normalized Gradients!
    """
    # Pad signals before STFT to preserve boundaries
    paddings_1d = tf.constant([[128, 128]])
    x_raw_padded = tf.pad(x_raw, paddings_1d, "REFLECT")
    y_raw_padded = tf.pad(y_raw, paddings_1d, "REFLECT")
    
    # tf.signal.stft computes over last axis dynamically. Expecting input shape [1280]
    stft_x = tf.signal.stft(x_raw_padded, frame_length=NPERSEG, frame_step=NOVERLAP, window_fn=tf.signal.hann_window)
    stft_y = tf.signal.stft(y_raw_padded, frame_length=NPERSEG, frame_step=NOVERLAP, window_fn=tf.signal.hann_window)
    
    # stft_x is [99, 129] purely native!
    # stft_x is [99, 129] purely native!
    x_real = tf.math.real(stft_x)
    x_imag = tf.math.imag(stft_x)
    y_real = tf.math.real(stft_y)
    y_imag = tf.math.imag(stft_y)
    
    # No padding strictly required since slice matching handles rounding natively
    paddings = tf.constant([[0, 0], [0, 0]]) # (time, freq)
    
    x_real_padded = tf.pad(x_real, paddings, "CONSTANT")
    x_imag_padded = tf.pad(x_imag, paddings, "CONSTANT")
    y_real_padded = tf.pad(y_real, paddings, "CONSTANT")
    y_imag_padded = tf.pad(y_imag, paddings, "CONSTANT")
    
    x_inp = tf.stack([x_real_padded, x_imag_padded], axis=-1)
    y_out = tf.stack([y_real_padded, y_imag_padded], axis=-1)
    
    # Normalizing functionally inside stream!
    # Sign-preserving log dynamically compresses range without NaN on negative values
    x_inp_norm = tf.math.sign(x_inp) * tf.math.log1p(tf.abs(x_inp))
    x_inp_norm = (x_inp_norm - APPROX_MEAN) / (APPROX_STD + 1e-8)
    
    y_out_norm = tf.math.sign(y_out) * tf.math.log1p(tf.abs(y_out))
    # Predict the symmetric scaled target!
    return x_inp_norm, y_out_norm

def create_tf_dataset():
    x_path = os.path.join(DS_DIR, "X_data.npy")
    y_path = os.path.join(DS_DIR, "Y_data.npy")
    
    print(f"Connecting memmap dynamically to {x_path}")
    X_mmap = np.lib.format.open_memmap(x_path, mode='r')
    Y_mmap = np.lib.format.open_memmap(y_path, mode='r')
    
    num_samples = X_mmap.shape[0]
    split_idx = int(num_samples * 0.8)
    print(f"Training exactly on {split_idx} natively streamed instances.")

    def gen_train():
        for i in range(split_idx):
            yield X_mmap[i, :, 0], Y_mmap[i, :, 0]
            
    def gen_test():
        for i in range(split_idx, num_samples):
            yield X_mmap[i, :, 0], Y_mmap[i, :, 0]

    output_signature = (tf.TensorSpec(shape=(1280,), dtype=tf.float32), 
                        tf.TensorSpec(shape=(1280,), dtype=tf.float32))

    train_ds = tf.data.Dataset.from_generator(gen_train, output_signature=output_signature)
    test_ds = tf.data.Dataset.from_generator(gen_test, output_signature=output_signature)

    # Parallel processing mappings (AUTOTUNE determines optimal threading dynamically)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(preprocess_tf_map, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=4096).repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    test_ds = test_ds.map(preprocess_tf_map, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    return train_ds, test_ds, split_idx, num_samples, X_mmap, Y_mmap

# -------------------------------------------------------------
# CRNN Architecture (Tan & Wang 2018)
# -------------------------------------------------------------
def build_crnn_model(time_steps, freq_bins):
    inputs = Input(shape=(time_steps, freq_bins, 2))
    
    def conv2d_block(x, filters):
        x = Conv2D(filters, kernel_size=(2, 3), strides=(1, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        x = Dropout(0.2)(x)
        return x

    def dec_comp(x, skip, filters):
        x = Conv2DTranspose(filters, kernel_size=(2, 3), strides=(1, 2), padding='same')(x)
        skip_f = tf.keras.backend.int_shape(skip)[2]
        x = x[:, :, :skip_f, :]
        x = Concatenate(axis=-1)([x, skip])
        x = BatchNormalization()(x)
        x = ELU()(x)
        x = Dropout(0.2)(x)
        return x

    # ENCODER
    c1 = conv2d_block(inputs, 16)   
    c2 = conv2d_block(c1, 32)       
    c3 = conv2d_block(c2, 64)       
    c4 = conv2d_block(c3, 128)      
    c5 = conv2d_block(c4, 256)      
    
    # BOTTLENECK (Group LSTM)
    # shape_before_lstm = tf.keras.backend.int_shape(c5)
    # b, t, f, c = shape_before_lstm
    # spatial_size = f * c
    # lstm_in = Reshape((time_steps, spatial_size))(c5)
    
    # def group_lstm_block(x, total_units, groups=2):
    #     from tensorflow.keras.layers import Lambda
    #     splits = Lambda(lambda f: tf.split(f, num_or_size_splits=groups, axis=-1))(x)
    #     lstm_outs = []
    #     for split_x in splits:
    #         lstm_outs.append(LSTM(total_units // groups, return_sequences=True)(split_x))
    #     return Concatenate(axis=-1)(lstm_outs)
    
    # lstm1 = group_lstm_block(lstm_in, spatial_size, groups=2)
    # lstm2 = group_lstm_block(lstm1, spatial_size, groups=2)
    
    # lstm_out = Reshape((time_steps, f, c))(lstm2)
    
    # DECODER
    #d4 = dec_comp(lstm_out, c4, 128)
    d4 = dec_comp(c5, c4, 128)
    d3 = dec_comp(d4, c3, 64)
    d2 = dec_comp(d3, c2, 32)
    d1 = dec_comp(d2, c1, 16)
    
    outputs = Conv2DTranspose(2, kernel_size=(2, 3), strides=(1, 2), padding='same', activation='linear')(d1)
    outputs = outputs[:, :, :freq_bins, :]
    
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# -------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------
class LivePlotCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        plt.figure()
        plt.plot(self.losses, label='Train Loss (MSE)')
        plt.plot(self.val_losses, label='Val Loss (MSE)')
        plt.title('CRNN Training Streaming Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('training_history_streaming.png')
        plt.close()

# -------------------------------------------------------------
# Main Execution Loop
# -------------------------------------------------------------
def main():
    print("Binding natively to physical disks...")
    train_ds, test_ds, split_idx, num_samples, X_mmap, Y_mmap = create_tf_dataset()
    
    # Explicit dims matched against TF `tf.signal.stft` calculations identically.
    time_steps = 23
    freq_bins = 65
    
    print("Building streaming CRNN...")
    model = build_crnn_model(time_steps, freq_bins)
    model.summary()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'anc_crnn_model.keras')
    model_weights_path = os.path.join(script_dir, 'anc_crnn_model.weights.h5')
    if os.path.exists(model_path):
        print(f"Model found at {model_path}. Loading weights directly from it to continue training...")
        model.load_weights(model_path)
    elif os.path.exists(model_weights_path):
        print(f"Weights found at {model_weights_path}. Loading weights to continue training...")
        model.load_weights(model_weights_path)
    else:
        print("No existing weights found, training from scratch...")
    
    # Steps dynamically computed via standard boundaries
    train_steps = split_idx // BATCH_SIZE
    val_steps = (num_samples - split_idx) // BATCH_SIZE
    
    print(f"Beginning asynchronous training sequence! Steps/Epoch: {train_steps}")
    live_plot = LivePlotCallback()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_ds, 
                        validation_data=test_ds,
                        epochs=EPOCHS, 
                        steps_per_epoch=train_steps,
                        validation_steps=val_steps,
                        callbacks=[live_plot, early_stopping])
    
    model.save_weights(os.path.join(script_dir, 'anc_crnn_model.weights.h5'))
    print("Model weights perfectly saved globally.")
    
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title('CRNN Training Streaming Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(script_dir, 'training_history_streaming.png'))
    plt.close()
    
    print("Evaluations logically flushed successfully.")

if __name__ == "__main__":
    main()
