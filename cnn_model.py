import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, ELU, Dropout
from tensorflow.keras.models import Model
import os
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend to avoid Tkinter thread loop errors
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Generic Configuration
# -------------------------------------------------------------
# Signal properties (Should match generate_dataset.py)
FS = 12800
DURATION = 0.1
RAW_SAMPLES = int(FS * DURATION) # 1280

# STFT Parameters (Fixed for 50Hz resolution to cleanly isolate 1500Hz)
NPERSEG = 256
NOVERLAP = 128
PAD_SAMPLES = NPERSEG  # Pad symmetrically by window length

# Dynamically calculate the STFT dimensions for the network input
PADDED_LENGTH = RAW_SAMPLES + 2 * PAD_SAMPLES
TIME_STEPS = 1 + (PADDED_LENGTH - NPERSEG) // NOVERLAP
FREQ_BINS = NPERSEG // 2 + 1

# Training Hyperparameters
EPOCHS = 20
BATCH_SIZE = 64

# --- MODEL CAPACITY CONTROLLER ---
# If you have a smaller dataset (e.g. < 50,000 samples), reduce BASE_FILTERS to 8 or 4
# to prevent the model from memorizing the noise (overfitting).
# For large datasets (e.g. 500,000+), 16 or 32 is appropriate.
BASE_FILTERS = 8 

script_dir = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(script_dir, "dataset_output")

# Normalization approximations
APPROX_MEAN = 0.0 
APPROX_STD = 1.0

# -------------------------------------------------------------
# Data Processing via tf.data
# -------------------------------------------------------------
def preprocess_tf_map(x_raw, y_raw):
    """
    Dynamically pads and converts raw signals to STFT domain with log1p normalization.
    """
    paddings_1d = tf.constant([[PAD_SAMPLES, PAD_SAMPLES]])
    x_raw_padded = tf.pad(x_raw, paddings_1d, "REFLECT")
    y_raw_padded = tf.pad(y_raw, paddings_1d, "REFLECT")
    
    stft_x = tf.signal.stft(x_raw_padded, frame_length=NPERSEG, frame_step=NOVERLAP, window_fn=tf.signal.hann_window)
    stft_y = tf.signal.stft(y_raw_padded, frame_length=NPERSEG, frame_step=NOVERLAP, window_fn=tf.signal.hann_window)
    
    x_inp = tf.stack([tf.math.real(stft_x), tf.math.imag(stft_x)], axis=-1)
    y_out = tf.stack([tf.math.real(stft_y), tf.math.imag(stft_y)], axis=-1)
    
    # Sign-preserving log dynamically compresses range without NaN
    x_inp_norm = tf.math.sign(x_inp) * tf.math.log1p(tf.abs(x_inp))
    x_inp_norm = (x_inp_norm - APPROX_MEAN) / (APPROX_STD + 1e-8)
    
    y_out_norm = tf.math.sign(y_out) * tf.math.log1p(tf.abs(y_out))
    
    return x_inp_norm, y_out_norm

def create_tf_dataset():
    x_path = os.path.join(DS_DIR, "X_data.npy")
    y_path = os.path.join(DS_DIR, "Y_data.npy")
    
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Dataset not found at {DS_DIR}. Run generate_dataset.py first.")
        
    print(f"Connecting memmap dynamically to {x_path}")
    X_mmap = np.lib.format.open_memmap(x_path, mode='r')
    Y_mmap = np.lib.format.open_memmap(y_path, mode='r')
    
    num_samples = X_mmap.shape[0]
    split_idx = int(num_samples * 0.8)
    print(f"Training on {split_idx} instances, Validating on {num_samples - split_idx} instances.")

    def gen_train():
        for i in range(split_idx):
            yield X_mmap[i, :, 0], Y_mmap[i, :, 0]
            
    def gen_test():
        for i in range(split_idx, num_samples):
            yield X_mmap[i, :, 0], Y_mmap[i, :, 0]

    output_signature = (tf.TensorSpec(shape=(RAW_SAMPLES,), dtype=tf.float32), 
                        tf.TensorSpec(shape=(RAW_SAMPLES,), dtype=tf.float32))

    train_ds = tf.data.Dataset.from_generator(gen_train, output_signature=output_signature)
    test_ds = tf.data.Dataset.from_generator(gen_test, output_signature=output_signature)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(preprocess_tf_map, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.shuffle(buffer_size=4096).repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    test_ds = test_ds.map(preprocess_tf_map, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    return train_ds, test_ds, split_idx, num_samples

# -------------------------------------------------------------
# Generic CNN Architecture (U-Net structure without recurrent bottlenecks)
# -------------------------------------------------------------
def build_cnn_model(time_steps, freq_bins, base_filters=16):
    inputs = Input(shape=(time_steps, freq_bins, 2))
    
    def conv2d_block(x, filters):
        x = Conv2D(filters, kernel_size=(2, 3), strides=(1, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        x = Dropout(0.2)(x)
        return x

    def dec_comp(x, skip, filters):
        x = Conv2DTranspose(filters, kernel_size=(2, 3), strides=(1, 2), padding='same')(x)
        # Slicing safely aligns dimensions in case of odd shape upsampling
        skip_f = tf.keras.backend.int_shape(skip)[2]
        x = x[:, :, :skip_f, :]
        x = Concatenate(axis=-1)([x, skip])
        x = BatchNormalization()(x)
        x = ELU()(x)
        x = Dropout(0.2)(x)
        return x

    # ENCODER
    c1 = conv2d_block(inputs, base_filters)           
    c2 = conv2d_block(c1, base_filters * 2)       
    c3 = conv2d_block(c2, base_filters * 4)       
    c4 = conv2d_block(c3, base_filters * 8)      
    c5 = conv2d_block(c4, base_filters * 16)      
    
    # DECODER
    d4 = dec_comp(c5, c4, base_filters * 8)
    d3 = dec_comp(d4, c3, base_filters * 4)
    d2 = dec_comp(d3, c2, base_filters * 2)
    d1 = dec_comp(d2, c1, base_filters)
    
    # Final projection to match input channels
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
        plt.title(f'CNN Training Streaming Loss (Base Filters: {BASE_FILTERS})')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('training_history_cnn.png')
        plt.close()

# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
def main():
    print("Initializing generic CNN for tone extraction...")
    print(f"Dynamically Calculated Network Input Shape: (Time={TIME_STEPS}, Freq={FREQ_BINS})")
    print(f"Base Filters capacity set to: {BASE_FILTERS}")
    
    train_ds, test_ds, split_idx, num_samples = create_tf_dataset()
    
    # Safely calculate steps to handle very small datasets dynamically
    train_steps = max(1, split_idx // BATCH_SIZE)
    val_steps = max(1, (num_samples - split_idx) // BATCH_SIZE)
    
    model = build_cnn_model(TIME_STEPS, FREQ_BINS, base_filters=BASE_FILTERS)
    model.summary()

    model_weights_path = os.path.join(script_dir, 'anc_cnn_model_best.weights.h5')
    
    if os.path.exists(model_weights_path):
        print(f"Weights found at {model_weights_path}. Loading to continue training...")
        model.load_weights(model_weights_path)
    else:
        print("No existing weights found, training from scratch...")
    
    # Callbacks configuration
    live_plot = LivePlotCallback()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    
    # ModelCheckpoint added so progress isn't lost on interruption
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_weights_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    print(f"Beginning training! Steps/Epoch: {train_steps}")
    
    history = model.fit(train_ds, 
                        validation_data=test_ds,
                        epochs=EPOCHS, 
                        steps_per_epoch=train_steps,
                        validation_steps=val_steps,
                        callbacks=[live_plot, early_stopping, checkpoint])
    
    # Fully save the completed model architecture and weights
    final_model_path = os.path.join(script_dir, 'anc_cnn_model_final.keras')
    model.save(final_model_path)
    print(f"Final model architecture and weights successfully saved to {final_model_path}")

if __name__ == "__main__":
    main()
