import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Generic Configuration
# -------------------------------------------------------------
FS = 12800
DURATION = 0.1
RAW_SAMPLES = int(FS * DURATION)  # 1280

NPERSEG = 256
NOVERLAP = 128
HOP_LENGTH = NPERSEG - NOVERLAP  # 128
PAD_SAMPLES = NPERSEG

PADDED_LENGTH = RAW_SAMPLES + 2 * PAD_SAMPLES
TIME_STEPS = 1 + (PADDED_LENGTH - NPERSEG) // NOVERLAP
FREQ_BINS = NPERSEG // 2 + 1

EPOCHS = 20
BATCH_SIZE = 64
BASE_FILTERS = 8

APPROX_MEAN = 0.0
APPROX_STD = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
DS_DIR = os.path.join(script_dir, "dataset_output")

# -------------------------------------------------------------
# Dataset
# -------------------------------------------------------------
class SignalDataset(Dataset):
    """
    Loads raw signals from .npy memmaps, pads and computes STFT on-the-fly,
    mirroring the TF preprocess_tf_map pipeline.

    STFT output shape: (TIME_STEPS, FREQ_BINS, 2)  [real + imag stacked last]
    """

    def __init__(self, x_mmap, y_mmap, indices):
        self.x_mmap = x_mmap
        self.y_mmap = y_mmap
        self.indices = indices

        # Pre-build a Hann window once and keep it on CPU;
        # it will be moved to DEVICE inside __getitem__ if needed,
        # but DataLoader workers share CPU so we stay on CPU here.
        self.window = torch.hann_window(NPERSEG)

    def __len__(self):
        return len(self.indices)

    def _to_stft(self, signal_1d: torch.Tensor) -> torch.Tensor:
        """
        Pads a 1-D signal with reflect padding then computes STFT.
        Returns a (TIME_STEPS, FREQ_BINS, 2) float32 tensor [real, imag].
        """
        # Reflect pad: torch.nn.functional.pad expects (..., left, right)
        padded = F.pad(signal_1d.unsqueeze(0).unsqueeze(0),
                       (PAD_SAMPLES, PAD_SAMPLES), mode='reflect').squeeze()

        # torch.stft returns (FREQ_BINS, TIME_STEPS, 2) with return_complex=False
        stft = torch.stft(
            padded,
            n_fft=NPERSEG,
            hop_length=HOP_LENGTH,
            win_length=NPERSEG,
            window=self.window,
            center=False,          # We already padded manually
            normalized=False,
            return_complex=False   # Returns (..., 2) real/imag
        )
        # stft: (FREQ_BINS, TIME_STEPS, 2) → transpose to (TIME_STEPS, FREQ_BINS, 2)
        stft = stft.permute(1, 0, 2).contiguous()
        return stft

    def _log_normalize(self, x: torch.Tensor, normalize_input=True) -> torch.Tensor:
        """
        Sign-preserving log1p compression matching TF pipeline.
        Optionally subtracts APPROX_MEAN and divides by APPROX_STD (for inputs only).
        """
        x = torch.sign(x) * torch.log1p(torch.abs(x))
        if normalize_input:
            x = (x - APPROX_MEAN) / (APPROX_STD + 1e-8)
        return x

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_raw = torch.from_numpy(self.x_mmap[real_idx, :, 0].astype(np.float32))
        y_raw = torch.from_numpy(self.y_mmap[real_idx, :, 0].astype(np.float32))

        x_stft = self._to_stft(x_raw)
        y_stft = self._to_stft(y_raw)

        x_norm = self._log_normalize(x_stft, normalize_input=True)
        y_norm = self._log_normalize(y_stft, normalize_input=False)

        # Channels-first for Conv2d: (2, TIME_STEPS, FREQ_BINS)
        x_norm = x_norm.permute(2, 0, 1).contiguous()
        y_norm = y_norm.permute(2, 0, 1).contiguous()

        return x_norm, y_norm


def create_dataloaders():
    x_path = os.path.join(DS_DIR, "X_data.npy")
    y_path = os.path.join(DS_DIR, "Y_data.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Dataset not found at {DS_DIR}. Run generate_dataset.py first.")

    print(f"Connecting memmap to {x_path}")
    X_mmap = np.lib.format.open_memmap(x_path, mode='r')
    Y_mmap = np.lib.format.open_memmap(y_path, mode='r')

    num_samples = X_mmap.shape[0]
    split_idx = int(num_samples * 0.8)
    print(f"Training on {split_idx} instances, Validating on {num_samples - split_idx} instances.")

    train_ds = SignalDataset(X_mmap, Y_mmap, list(range(split_idx)))
    val_ds   = SignalDataset(X_mmap, Y_mmap, list(range(split_idx, num_samples)))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True, drop_last=True)

    return train_loader, val_loader, split_idx, num_samples


# -------------------------------------------------------------
# U-Net Model
# -------------------------------------------------------------
class EncoderBlock(nn.Module):
    """
    Conv2d with stride-2 downsampling along frequency axis,
    BatchNorm, ELU, and Dropout — matches the TF conv2d_block.
    """
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        # kernel (2,3), stride (1,2): time unchanged, freq halved
        self.conv  = nn.Conv2d(in_ch, out_ch, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))
        self.bn    = nn.BatchNorm2d(out_ch)
        self.act   = nn.ELU()
        self.drop  = nn.Dropout2d(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class DecoderBlock(nn.Module):
    """
    ConvTranspose2d upsampling, skip-connection concatenation,
    BatchNorm, ELU, Dropout — matches the TF dec_comp block.
    """
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.2):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))
        # After concat with skip: channels = out_ch + skip_ch
        self.bn   = nn.BatchNorm2d(out_ch + skip_ch)
        self.act  = nn.ELU()
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Trim frequency dim to match skip (handles odd shapes from strided conv)
        if x.shape[-1] != skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)   # concat along channel dim
        return self.drop(self.act(self.bn(x)))


class UNetCNN(nn.Module):
    """
    5-level U-Net with strided Conv2d encoder and ConvTranspose2d decoder.
    Input/output shape: (B, 2, TIME_STEPS, FREQ_BINS)
    """
    def __init__(self, base_filters=16):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = EncoderBlock(2,       f)
        self.enc2 = EncoderBlock(f,       f * 2)
        self.enc3 = EncoderBlock(f * 2,   f * 4)
        self.enc4 = EncoderBlock(f * 4,   f * 8)
        self.enc5 = EncoderBlock(f * 8,   f * 16)   # bottleneck

        # Decoder  (in_ch = bottleneck channels, skip_ch = encoder skip channels)
        self.dec4 = DecoderBlock(f * 16, f * 8,  f * 8)
        self.dec3 = DecoderBlock(f * 8  + f * 8,  f * 4,  f * 4)
        self.dec2 = DecoderBlock(f * 4  + f * 4,  f * 2,  f * 2)
        self.dec1 = DecoderBlock(f * 2  + f * 2,  f,      f)

        # Final output projection
        self.out_conv = nn.ConvTranspose2d(f + f, 2, kernel_size=(2, 3),
                                           stride=(1, 2), padding=(0, 1))

    def forward(self, x):
        # Encoder path
        c1 = self.enc1(x)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        c5 = self.enc5(c4)   # bottleneck

        # Decoder path with skip connections
        d4 = self.dec4(c5, c4)
        d3 = self.dec3(d4, c3)
        d2 = self.dec2(d3, c2)
        d1 = self.dec1(d2, c1)

        # Final upsample + trim to exact FREQ_BINS
        out = self.out_conv(d1)
        if out.shape[-1] != FREQ_BINS:
            out = out[..., :FREQ_BINS]

        return out


# -------------------------------------------------------------
# Training Utilities
# -------------------------------------------------------------
def save_loss_plot(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses,   label='Val Loss (MSE)')
    plt.title(f'CNN Training Loss (Base Filters: {BASE_FILTERS})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(script_dir, 'training_history_cnn.png'))
    plt.close()


def run_epoch(model, loader, optimizer, criterion, train=True):
    model.train(train)
    total_loss = 0.0
    steps = 0
    with torch.set_grad_enabled(train):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            steps += 1

    return total_loss / max(steps, 1)


# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
def main():
    print(f"Using device: {DEVICE}")
    print(f"Input shape  : (2, {TIME_STEPS}, {FREQ_BINS})")
    print(f"Base Filters : {BASE_FILTERS}")

    train_loader, val_loader, split_idx, num_samples = create_dataloaders()

    model     = UNetCNN(base_filters=BASE_FILTERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Print parameter count (mirrors model.summary())
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    weights_path = os.path.join(script_dir, 'anc_cnn_model_best.pt')
    final_path   = os.path.join(script_dir, 'anc_cnn_model_final.pt')

    if os.path.exists(weights_path):
        print(f"Checkpoint found at {weights_path}. Loading to continue training...")
        checkpoint = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch      = checkpoint.get('epoch', 0)
        best_val_loss    = checkpoint.get('best_val_loss', float('inf'))
        train_losses     = checkpoint.get('train_losses', [])
        val_losses       = checkpoint.get('val_losses', [])
    else:
        print("No checkpoint found. Training from scratch...")
        start_epoch   = 0
        best_val_loss = float('inf')
        train_losses  = []
        val_losses    = []

    patience        = 4
    patience_counter = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(start_epoch, EPOCHS):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, train=True)
        val_loss   = run_epoch(model, val_loader,   optimizer, criterion, train=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch + 1:02d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch':           epoch + 1,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_val_loss':   best_val_loss,
                'train_losses':    train_losses,
                'val_losses':      val_losses,
            }, weights_path)
            print(f"  → Best model saved (val_loss={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        save_loss_plot(train_losses, val_losses)

    # Restore best weights before final save (mirrors restore_best_weights=True)
    best_ckpt = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(best_ckpt['model_state'])

    torch.save(model.state_dict(), final_path)
    print(f"Final model weights saved to {final_path}")


if __name__ == "__main__":
    main()