import os
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
    Reads wav, returns (T, 80) log10-mel spectrogram.
    """
    wav, sr = sf.read(wav_path)
    if wav.ndim == 2:
        wav = wav.mean(axis=1) # mono
    wav = wav.astype(np.float32)

    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

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

    return logmel.T.astype(np.float32) # Return (T, 80)

def logmel_to_image(logmel, min_val=MIN_DB, max_val=MAX_DB):
    """
    Converts (T, 80) float logmel to PIL Image (Grayscale).
    """
    # Normalize to 0-1
    norm = (logmel - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0.0, 1.0)
    
    # Scale to 0-255
    uint8_data = (norm * 255.0).astype(np.uint8)
    
    img_data = np.flipud(uint8_data.T) # (80, T) -> freq is height, time is width
    
    return Image.fromarray(img_data, mode='L')

def image_to_logmel(image, min_val=MIN_DB, max_val=MAX_DB):
    """
    Converts PIL Image to (T, 80) logmel.
    """
    image = image.convert('L')
    img_data = np.array(image).astype(np.float32)
    
    # Flip back
    img_data = np.flipud(img_data)
    
    # Transpose back: (80, T) -> (T, 80)
    logmel_norm = img_data.T / 255.0
    
    # De-normalize
    logmel = logmel_norm * (max_val - min_val) + min_val
    
    return logmel

def reconstruct_wav(logmel, vocoder, device):
    """
    logmel: (T, 80)
    """
    spectrogram = torch.tensor(logmel).unsqueeze(0).to(device)
    
    with torch.no_grad():
        waveform = vocoder(spectrogram)
        
    return waveform.squeeze().cpu().numpy()
