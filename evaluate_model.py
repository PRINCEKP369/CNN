import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

from cnn_model import build_cnn_model, TIME_STEPS, FREQ_BINS, BASE_FILTERS, NPERSEG, NOVERLAP, PAD_SAMPLES, FS

def denormalize_and_istft(normalized_spectrogram):
    # Invert log1p dynamically
    spectrogram = tf.math.sign(normalized_spectrogram) * tf.math.expm1(tf.math.abs(normalized_spectrogram))
    
    # Create complex spectrogram
    c_spectrogram = tf.complex(spectrogram[..., 0], spectrogram[..., 1])
    
    # ISTFT
    inverse_stft_window = tf.signal.inverse_stft_window_fn(NOVERLAP, forward_window_fn=tf.signal.hann_window)
    time_domain = tf.signal.inverse_stft(c_spectrogram, frame_length=NPERSEG, frame_step=NOVERLAP, window_fn=inverse_stft_window)
    
    return time_domain

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'anc_cnn_model_best.weights.h5')
    print(f"Loading weights from {model_path} into recreated model...")
    model = build_cnn_model(TIME_STEPS, FREQ_BINS, base_filters=BASE_FILTERS)
    model.load_weights(model_path)
    
    x_path = os.path.join(script_dir, "dataset_output", "X_data.npy")
    y_path = os.path.join(script_dir, "dataset_output", "Y_data.npy")
    
    print("Loading data...")
    X_mmap = np.lib.format.open_memmap(x_path, mode='r')
    Y_mmap = np.lib.format.open_memmap(y_path, mode='r')
    
    # Take a sample from the test split
    TOTAL = X_mmap.shape[0]
    test_idx = int(TOTAL * 0.8) + 123 # Arbitrary test index
    
    x_sample = X_mmap[test_idx, :, 0]
    y_sample_true = Y_mmap[test_idx, :, 0]
    
    # Apply identical STFT and normalizations to the X data
    x_tensor = tf.convert_to_tensor([x_sample], dtype=tf.float32)
    paddings_1d = tf.constant([[0, 0], [PAD_SAMPLES, PAD_SAMPLES]])
    x_tensor_padded = tf.pad(x_tensor, paddings_1d, "REFLECT")
    stft_x = tf.signal.stft(x_tensor_padded, frame_length=NPERSEG, frame_step=NOVERLAP, window_fn=tf.signal.hann_window)
    x_real = tf.math.real(stft_x)
    x_imag = tf.math.imag(stft_x)
    x_inp = tf.stack([x_real, x_imag], axis=-1)
    
    # Apply log1p transform natively
    x_inp_norm = tf.math.sign(x_inp) * tf.math.log1p(tf.math.abs(x_inp))
    x_inp_norm = (x_inp_norm - 0.0) / (1.0 + 1e-8)
    
    print("Making prediction...")
    # Predict output spectrum
    y_pred_norm = model.predict(x_inp_norm)
    
    # Noise Gate: Clamp out faint NN mathematical fuzz predicting near absolute zero ambient noise
    y_pred_norm = tf.where(tf.math.abs(y_pred_norm) < 0.01, tf.zeros_like(y_pred_norm), y_pred_norm)
    
    # Inverse map to time domain
    y_pred_time_padded = denormalize_and_istft(y_pred_norm)[0].numpy()
    
    # Trim the padding
    y_pred_time = y_pred_time_padded[PAD_SAMPLES:-PAD_SAMPLES]
    
    # Ensure same length 
    min_len = min(len(x_sample), len(y_pred_time))
    x_sample = x_sample[:min_len]
    y_sample_true = y_sample_true[:min_len]
    y_pred_time = y_pred_time[:min_len]
    
    time_axis = np.arange(min_len) / FS
    
    print("Saving plots...")
    # Plot Time Domain
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, x_sample, label='Noisy Input (X)', color='orange', alpha=0.9)
    plt.title('Time Domain - Noisy Input (Target Interference + 1500 Hz Towship)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, y_sample_true, label='Clean Target', color='blue', alpha=0.7)
    plt.plot(time_axis, y_pred_time, label='Predicted Target', color='green', alpha=0.7, linestyle='--')
    plt.title('Time Domain - Clean & Predicted Overlapped')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'result_time_domain.png'), dpi=150)
    plt.close()
    
    # Plot Frequency Domain
    xf = rfftfreq(min_len, 1 / FS)
    yf_x = rfft(x_sample)
    yf_y_true = rfft(y_sample_true)
    yf_y_pred = rfft(y_pred_time)
    
    # mag_x = 20 * np.log10(np.abs(yf_x) + 1e-12)
    # mag_y_true = 20 * np.log10(np.abs(yf_y_true) + 1e-12)
    # mag_y_pred = 20 * np.log10(np.abs(yf_y_pred) + 1e-12)
    
    mag_x = np.abs(yf_x)
    mag_y_true = np.abs(yf_y_true)
    mag_y_pred = np.abs(yf_y_pred)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(xf, mag_x, label='Noisy Input', color='orange')
    plt.title('Frequency Domain - Noisy Input')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.xlim(1000, 2200)
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(xf, mag_y_true, label='Clean Target', color='blue', alpha=0.7)
    plt.plot(xf, mag_y_pred, label='Predicted Output', color='green', alpha=0.7, linestyle='--')
    plt.title('Frequency Domain - Clean & Predicted Overlapped')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.xlim(1000, 2200)
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'result_frequency_domain.png'), dpi=150)
    plt.close()
    
    print("Saved result_time_domain.png and result_frequency_domain.png.")

if __name__ == '__main__':
    main()
