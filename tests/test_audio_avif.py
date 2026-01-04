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
            try:
                import shutil
                shutil.rmtree(self.test_dir)
            except:
                pass

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

    def test_reshape_logic(self):
        # Generate longer audio (5s) to trigger reshaping
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        long_wav_path = os.path.join(self.test_dir, "long_tone.wav")
        sf.write(long_wav_path, audio, sr)
        
        logmel, rms_orig = audio_avif.wav_to_logmel(long_wav_path)
        T_orig = logmel.shape[0] # (T, 80)
        
        # 1. Test WITH Reshaping (Explicit)
        img_reshaped = audio_avif.logmel_to_image(logmel, rms=rms_orig, reshape=True)
        w, h = img_reshaped.size
        
        # Expect roughly square
        # T ~ 313 frames. 80*313 = 25040 pixels. sqrt ~ 158.
        # k = round(sqrt(313/80)) = round(1.97) = 2.
        # Height should be 80 * 2 = 160.
        self.assertEqual(h % 80, 0, "Height should be multiple of 80")
        self.assertTrue(h > 80, "Should have stacked at least 2 strips for 5s audio")
        
        # Check metadata
        exif = img_reshaped.getexif()
        desc = exif.get(270)
        self.assertIn("orig_w", desc, "Metadata should contain original width")
        
        # Reconstruction
        logmel_recon, rms_recon = audio_avif.image_to_logmel(img_reshaped)
        self.assertEqual(logmel_recon.shape, logmel.shape, "Reconstructed shape must match original")
        
        # 2. Test WITHOUT Reshaping
        img_linear = audio_avif.logmel_to_image(logmel, rms=rms_orig, reshape=False)
        w_lin, h_lin = img_linear.size
        self.assertEqual(h_lin, 80, "Linear height must be 80")
        self.assertEqual(w_lin, T_orig, "Linear width must match time frames")
        
        # Reconstruction
        logmel_recon_lin, _ = audio_avif.image_to_logmel(img_linear)
        self.assertEqual(logmel_recon_lin.shape, logmel.shape)
        
        os.remove(long_wav_path)

if __name__ == '__main__':
    unittest.main()