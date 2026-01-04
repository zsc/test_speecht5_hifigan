import os
import math
import json
import numpy as np
import soundfile as sf
import librosa
import torch
import torchaudio
from transformers import SpeechT5HifiGan
from PIL import Image
import pillow_avif

# Constants
TARGET_SR = 16000
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 80
FMAX = 7600
MIN_DB = -11.0
MAX_DB = 4.0
QUALITIES = [70, 80, 85, 90, 95]

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_vocoder(device=None):
    if device is None:
        device = get_device()
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    vocoder.eval()
    return vocoder

def wav_to_logmel(wav_path):
    """
    Reads wav, returns ((T, 80) log10-mel spectrogram, rms).
    rms is calculated on the 16kHz audio used for mel extraction.
    """
    wav, sr = sf.read(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1) # mono
    wav = wav.astype(np.float32)

    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    
    # Calculate RMS
    rms = np.sqrt(np.mean(wav**2))

    # Compute STFT
    S = np.abs(
        librosa.stft(
            wav,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window="hann",
            center=True,
            pad_mode="reflect",
        )
    )

    # Mel basis
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=False,         # Slaney
        norm="slaney",     # Slaney
    )

    mel = mel_basis @ S
    mel = np.maximum(mel, 1e-10)
    logmel = np.log10(mel) # (80, T)

    return logmel.T.astype(np.float32), float(rms) # Return (T, 80), rms

def logmel_to_image(logmel, min_val=MIN_DB, max_val=MAX_DB, rms=None, reshape=False):
    """
    Converts (T, 80) float logmel to PIL Image (Grayscale).
    
    Args:
        reshape (bool): If True, reshapes long spectrograms into a roughly square image
                        by stacking time slices vertically. This often improves compression.
    """
    # Normalize to 0-1
    norm = (logmel - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0.0, 1.0)
    
    # Scale to 0-255
    uint8_data = (norm * 255.0).astype(np.uint8)
    
    # Standard spectrogram orientation: Frequency is Y-axis (height), Time is X-axis (width)
    # logmel is (Time, Freq) -> (T, 80)
    # We transpose to (Freq, Time) -> (80, T)
    # And flipud so low freq is at bottom
    img_data = np.flipud(uint8_data.T) # (80, T)
    
    original_width = img_data.shape[1]
    metadata = {}
    
    if rms is not None:
        metadata['rms'] = rms

    if reshape:
        # Square heuristic
        T = original_width
        # Target roughly square: Side ~ sqrt(80 * T)
        # Number of strips k = Side / 80 = sqrt(T/80)
        k = max(1, int(round(math.sqrt(T / 80.0))))
        
        # Calculate width per strip
        width_per_strip = math.ceil(T / k)
        
        # Align to 16 for compression block efficiency
        if width_per_strip % 16 != 0:
            width_per_strip = ((width_per_strip // 16) + 1) * 16
            
        total_width_needed = width_per_strip * k
        pad_amount = total_width_needed - T
        
        if pad_amount > 0:
            # Pad with 0 (silence equivalent in normalized space)
            padding = np.zeros((80, pad_amount), dtype=np.uint8)
            img_data = np.hstack([img_data, padding])
            
        # Split and Stack Vertically
        # Each chunk is (80, width_per_strip)
        chunks = [img_data[:, i*width_per_strip : (i+1)*width_per_strip] for i in range(k)]
        img_data = np.vstack(chunks) # (80*k, width_per_strip)
        
        metadata['orig_w'] = original_width

    img = Image.fromarray(img_data, mode='L')
    
    if metadata:
        exif = img.getexif()
        # Tag 270 is ImageDescription
        exif[270] = json.dumps(metadata)
        # Note: The caller must use img.save(..., exif=img.getexif())
        # We attach it to the image instance for convenience if supported
        pass
        
    return img

def image_to_logmel(image, min_val=MIN_DB, max_val=MAX_DB):
    """
    Converts PIL Image to (T, 80) logmel.
    Handles un-reshaping if metadata indicates the image was squared.
    Returns (logmel, rms). rms is None if not found in metadata.
    """
    # Parse Metadata
    rms = None
    orig_w = None
    
    exif = image.getexif()
    if exif and 270 in exif:
        desc = exif[270]
        # Try Parsing JSON
        try:
            meta = json.loads(desc)
            if isinstance(meta, dict):
                rms = meta.get('rms')
                orig_w = meta.get('orig_w')
        except json.JSONDecodeError:
            # Fallback to legacy format "OriginalRMS:0.123"
            if isinstance(desc, str) and desc.startswith("OriginalRMS:"):
                try:
                    rms = float(desc.split(":")[1])
                except:
                    pass

    image = image.convert('L')
    img_data = np.array(image) # (H, W)
    
    # Un-reshape if needed
    if orig_w is not None:
        H, W = img_data.shape
        # We know each strip is 80 pixels high
        if H % 80 == 0:
            k = H // 80
            # Split vertically
            chunks = np.vsplit(img_data, k)
            # Stack horizontally
            img_data = np.hstack(chunks) # (80, k*W)
            # Crop padding
            img_data = img_data[:, :orig_w]
        else:
            print(f"Warning: Image height {H} is not a multiple of 80, cannot un-reshape correctly. Treating as standard.")

    img_data = img_data.astype(np.float32)
    
    # Flip back (Spectrogram was flipud)
    img_data = np.flipud(img_data)
    
    # Transpose back: (80, T) -> (T, 80)
    logmel_norm = img_data.T / 255.0
    
    # De-normalize
    logmel = logmel_norm * (max_val - min_val) + min_val
    
    return logmel, rms

def reconstruct_wav(logmel, vocoder, device):
    """
    logmel: (T, 80)
    """
    spectrogram = torch.tensor(logmel).unsqueeze(0).to(device)
    
    with torch.no_grad():
        waveform = vocoder(spectrogram)
        
    return waveform.squeeze().cpu().numpy()

def apply_loudness(wav, target_rms):
    """
    Adjusts wav loudness to match target_rms.
    """
    if target_rms is None:
        return wav
        
    current_rms = np.sqrt(np.mean(wav**2))
    if current_rms <= 1e-9:
        return wav
        
    gain = target_rms / current_rms
    return wav * gain