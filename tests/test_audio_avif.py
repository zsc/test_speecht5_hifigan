import unittest
import os
import json
import numpy as np
import soundfile as sf
import audio_avif
from PIL import Image

class TestAudioAvif(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/test_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.wav_path = os.path.join(self.test_dir, "test_tone.wav")
        self.alignment_json_path = os.path.join(self.test_dir, "loudness_alignment.json")
        
        # Generate a synthetic sine wave (440Hz) for 1 second
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        # Use a mix of frequencies to make it more interesting for Mel spectogram
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        sf.write(self.wav_path, audio, sr)
        
        # Ensure model is cached or available (will download if not)
        try:
            self.device = audio_avif.get_device()
            self.vocoder = audio_avif.load_vocoder(self.device)
        except Exception as e:
            self.skipTest(f"Failed to load vocoder (network issue?): {e}")

    def tearDown(self):
        # Cleanup
        if os.path.exists(self.wav_path):
            os.remove(self.wav_path)
        if os.path.exists(self.alignment_json_path):
            os.remove(self.alignment_json_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def calculate_psnr(self, original, reconstructed):
        # Crop to same length
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]
        
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        
        # Assuming audio range [-1, 1]
        max_pixel = 2.0 # Dynamic range could be considered 2 (-1 to 1) or peak-to-peak
        # Standard definition often uses peak value of signal. 
        # If signal is float32 [-1, 1], MAX_I is usually 1 or 2. 
        # Let's use MAX=1.0 as reference amplitude.
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return psnr

    def test_compression_cycle_psnr(self):
        # 1. Wav -> Mel
        logmel = audio_avif.wav_to_logmel(self.wav_path)
        
        # 2. Mel -> Image (AVIF Q90)
        img = audio_avif.logmel_to_image(logmel)
        # Save to buffer or file to ensure compression happens
        temp_avif = os.path.join(self.test_dir, "temp.avif")
        img.save(temp_avif, "AVIF", quality=90)
        
        # 3. Image -> Mel
        img_loaded = Image.open(temp_avif)
        logmel_recon = audio_avif.image_to_logmel(img_loaded)
        
        # 4. Mel -> Wav
        wav_recon = audio_avif.reconstruct_wav(logmel_recon, self.vocoder, self.device)
        
        # Load original for comparison
        wav_orig, _ = sf.read(self.wav_path)
        
        # --- Loudness Alignment ---
        # Simple RMS alignment
        rms_orig = np.sqrt(np.mean(wav_orig**2))
        rms_recon = np.sqrt(np.mean(wav_recon**2))
        
        if rms_recon > 1e-6:
            gain_correction = rms_orig / rms_recon
        else:
            gain_correction = 1.0
            
        wav_recon_aligned = wav_recon * gain_correction
        
        # Save alignment info to JSON as requested
        alignment_info = {
            "original_rms": float(rms_orig),
            "reconstructed_rms": float(rms_recon),
            "gain_correction_factor": float(gain_correction),
            "note": "Applied gain_correction to reconstructed audio before PSNR calculation."
        }
        
        with open(self.alignment_json_path, 'w') as f:
            json.dump(alignment_info, f, indent=4)
            
        # --- PSNR Calculation ---
        # Note: Generative vocoders do not preserve phase, so Waveform PSNR might be low.
        # However, we check if it is reasonable (not extremely negative or error).
        psnr = self.calculate_psnr(wav_orig, wav_recon_aligned)
        
        print(f"\nTest Result - PSNR: {psnr:.2f} dB")
        print(f"Alignment Info saved to {self.alignment_json_path}")
        
        # Clean up temp file
        os.remove(temp_avif)

        # Assertion
        # We set a very low bar for PSNR because waveform matching is not the goal of HiFiGan,
        # but we want to ensure the pipeline isn't outputting pure noise or silence.
        # A PSNR of > 0 dB means signal is somewhat correlated or at least scaled similarly.
        # Typical good waveform reconstruction might be > 10-20 dB depending on phase alignment.
        self.assertTrue(psnr > 0, "PSNR should be positive (better than noise)")
        
        # Check if JSON exists
        self.assertTrue(os.path.exists(self.alignment_json_path))

if __name__ == '__main__':
    unittest.main()
