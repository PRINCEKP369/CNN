import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import os

# -------------------------------------------------------------
# Configuration & Parameters
# -------------------------------------------------------------
FS = 12800            # Sampling frequency (Hz)
DURATION = 0.1       # Signal duration (s)
N_SAMPLES = int(FS * DURATION)
TOWSHIP_FREQ = 1500   # Towship tonal frequency (Hz)
TOWSHIP_SNR_DB = 10.0 # Fixed SNR for towship vs radiated noise
TARGET_SNR=10 # Fixed SNR for target vs radiated noise
# Frequencies for target (interference): 1100 to 1950 step 50, excluding 1500
TARGET_FREQS = [f for f in range(1100, 2000, 50) if f != 1500]

# Two sets of parameters requested by the user:
TARGET_SNRS_DB = [0,5,10]
SINRS_DB = [-8,-5,-3,-2]

TOTAL_SCENARIOS = 100000

# Directory to save dataset
DS_DIR = "dataset_output"
os.makedirs(DS_DIR, exist_ok=True)

# -------------------------------------------------------------
# Signal Generation Functions
# -------------------------------------------------------------
def generate_tone(freq, fs, num_samples):
    """Generate a single sine wave tone."""
    t = np.arange(num_samples) / fs
    return np.sin(2 * np.pi * freq * t )

def calculate_power(signal):
    """Calculate the average power of a signal."""
    return np.mean(signal**2)

def generate_scenario(target_freq, target_snr_db, sinr_db):
    """
    Generates a single synthetic acoustic scenario.
    Because SINR = SNR_tow - SNR_tgt, we cannot define target_snr_db AND sinr_db 
    independently if SNR_tow is fixed at 10 dB. 
    Therefore, we pass either target_snr_db OR sinr_db to define the target power.
    """
    # 1. Generate Gaussian White Noise (towship radiated noise)
    noise = np.random.randn(N_SAMPLES)
    noise_power = calculate_power(noise)
    
    # 2. Generate Towship Signal (Desired Signal)
    #towship_phase = np.random.uniform(0, 2 * np.pi)
    towship_signal_raw = generate_tone(TOWSHIP_FREQ, FS, N_SAMPLES)
    
    desired_towship_power = noise_power * (10 ** (TOWSHIP_SNR_DB / 10.0))
    current_towship_power = calculate_power(towship_signal_raw)
    towship_signal = towship_signal_raw * np.sqrt(desired_towship_power / current_towship_power)+noise
    
    # 3. Generate Target Signal 
    noise1 = np.random.randn(N_SAMPLES)
    noise_power1 = calculate_power(noise1)
    #target_phase = np.random.uniform(0, 2 * np.pi)
    target_signal_raw = generate_tone(target_freq, FS, N_SAMPLES)
    current_target_power = calculate_power(target_signal_raw)
    
    desired_target_power = noise_power1 * (10** (TARGET_SNR / 10.0))
    target_signal = target_signal_raw * np.sqrt(desired_target_power / current_target_power)+noise1

    target_power=calculate_power(target_signal)#power without adding ambient noise
    towship_power=calculate_power(towship_signal)
    

    #4.Generate ambient noise
    ambient_noise = np.random.randn(N_SAMPLES)#ambient noise
    ambient_noise_power = calculate_power(ambient_noise)

    desired_targetsig_pow= ambient_noise_power * (10 ** (target_snr_db/ 10.0))#desired target signal power with respect to ambient noise
    target_signal_new=target_signal* np.sqrt(desired_targetsig_pow / target_power)
    #SINR SCALING
    alpha=((target_power/(10**(sinr_db/10.0)))-ambient_noise_power)/towship_power
    noisy_input= alpha*towship_signal + target_signal_new

    return noisy_input, alpha*towship_signal

# -------------------------------------------------------------
# Plotting Utility
# -------------------------------------------------------------
def plot_fft(noisy_input, clean_output, filename="fft_plot.png"):
    """Plot FFT of the noisy input and the clean towship output."""
    yf_noisy = np.fft.fftshift(np.fft.fft(noisy_input))
    yf_clean = np.fft.fftshift(np.fft.fft(clean_output))
    xf = np.fft.fftshift(np.fft.fftfreq(N_SAMPLES, 1 / FS))
    
    # Convert to Magnitude (dB scale for better visibility)
    # mag_noisy = 20 * np.log10(np.abs(yf_noisy))
    # mag_clean = 20 * np.log10(np.abs(yf_clean))
    
    mag_noisy = np.abs(yf_noisy)
    mag_clean = np.abs(yf_clean)
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Noisy Input FFT
    plt.subplot(2, 1, 1)
    plt.plot(xf,mag_noisy, color='r')
    plt.title("FFT of Noisy Input Signal (Towship + Target)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
   # plt.xlim(1000, 2200) # Zoom into relevant range
    
    # Subplot 2: Clean Output FFT
    plt.subplot(2, 1, 2)
    plt.plot(xf,mag_clean, color='b')
    plt.title("FFT of Output Signal (Clean Towship Signal Only)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
   # plt.xlim(1000, 2200) # Zoom into relevant range
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# -------------------------------------------------------------
# Main Generation Loop
# -------------------------------------------------------------
def main():
    print(f"Generating dataset of {TOTAL_SCENARIOS} samples...")
    
    out_x_path = os.path.join(DS_DIR, "X_data.npy")
    out_y_path = os.path.join(DS_DIR, "Y_data.npy")
    
    # Memory mapping is CRUCIAL here. A normal 100K batch would be ~10 GB
    # and crash your python process.
    X_array = np.lib.format.open_memmap(out_x_path, mode='w+', dtype=np.float32, shape=(TOTAL_SCENARIOS, N_SAMPLES, 1))
    Y_array = np.lib.format.open_memmap(out_y_path, mode='w+', dtype=np.float32, shape=(TOTAL_SCENARIOS, N_SAMPLES, 1))
    
    plotted = False
    
    for i in range(TOTAL_SCENARIOS):
        f = np.random.choice(TARGET_FREQS)
        t_snr = np.random.choice(TARGET_SNRS_DB)
        sinr = np.random.choice(SINRS_DB)
        
        noisy_in, clean_out = generate_scenario(target_freq=f, target_snr_db=t_snr, sinr_db=sinr)
        
        X_array[i, :, 0] = noisy_in
        Y_array[i, :, 0] = clean_out
        
        if not plotted:
            plot_fft(noisy_in, clean_out, os.path.join(DS_DIR, "fft_sample_target_snr.png"))
            plotted = True
            
        if (i + 1) % 5000 == 0:
            print(f"Generated {i + 1} / {TOTAL_SCENARIOS} scenarios...")
            plot_fft(noisy_in, clean_out, os.path.join(DS_DIR, f"fft_sample_{i+1}.png"))
            # Flush changes to disk intermittently
            X_array.flush()
            Y_array.flush()
            
    print(f"Dataset generated completely. Total scenarios: {X_array.shape[0]}")
    print(f"X shape: {X_array.shape}")
    print(f"Y shape: {Y_array.shape}")
    print(f"Saved disk-backed X to {out_x_path}")
    print(f"Saved disk-backed Y to {out_y_path}")

if __name__ == "__main__":
    main()
