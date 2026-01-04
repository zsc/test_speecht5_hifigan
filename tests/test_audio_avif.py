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
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
        sf.write(self.wav_path, audio, sr)
        
        try:
            self.device = audio_avif.get_device()
            self.vocoder = audio_avif.load_vocoder(self.device)
        except Exception as e:
            self.skipTest(f"Failed to load vocoder: {e}")

    def tearDown(self):
        if os.path.exists(self.wav_path):
            os.remove(self.wav_path)
        if os.path.exists(self.alignment_json_path):
            os.remove(self.alignment_json_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)

    def calculate_psnr(self, original, reconstructed):
        min_len = min(len(original), len(reconstructed))
        original = original[:min_len]
        reconstructed = reconstructed[:min_len]
        
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return psnr

    def test_compression_cycle_psnr(self):
        # 1. Wav -> Mel (with RMS)
        logmel, rms_orig = audio_avif.wav_to_logmel(self.wav_path)
        
        # 2. Mel -> Image (AVIF Q90, embed RMS)
        img = audio_avif.logmel_to_image(logmel, rms=rms_orig)
        temp_avif = os.path.join(self.test_dir, "temp.avif")
        img.save(temp_avif, "AVIF", quality=90, exif=img.getexif())
        
        # 3. Image -> Mel (Extract RMS)
        img_loaded = Image.open(temp_avif)
        logmel_recon, rms_recon = audio_avif.image_to_logmel(img_loaded)
        
        self.assertIsNotNone(rms_recon, "RMS metadata should be recovered from AVIF")
        self.assertAlmostEqual(rms_orig, rms_recon, places=4, msg="Recovered RMS should match original")
        
        # 4. Mel -> Wav
        wav_recon = audio_avif.reconstruct_wav(logmel_recon, self.vocoder, self.device)
        
        # Apply loudness correction
        wav_recon_aligned = audio_avif.apply_loudness(wav_recon, rms_recon)
        
        # Load original for comparison
        wav_orig, _ = sf.read(self.wav_path)
        
        # --- Stats for JSON ---
        rms_final = np.sqrt(np.mean(wav_recon_aligned**2))
        
        alignment_info = {
            "original_rms": float(rms_orig),
            "reconstructed_rms_before_align": float(np.sqrt(np.mean(wav_recon**2))),
            "reconstructed_rms_after_align": float(rms_final),
            "metadata_rms": float(rms_recon) if rms_recon else None,
            "note": "RMS stored in Exif/ImageDescription and applied."
        }
        
        with open(self.alignment_json_path, 'w') as f:
            json.dump(alignment_info, f, indent=4)
            
        psnr = self.calculate_psnr(wav_orig, wav_recon_aligned)
        
        print(f"\nTest Result - PSNR: {psnr:.2f} dB")
        print(f"Alignment Metadata Check: Orig={rms_orig:.4f}, Recon={rms_recon:.4f}")
        
        os.remove(temp_avif)

        self.assertTrue(psnr > 0, "PSNR should be positive")

if __name__ == '__main__':
    unittest.main()