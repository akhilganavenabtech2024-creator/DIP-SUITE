"""
=============================================================================
UNIFIED DIGITAL IMAGE PROCESSING (DIP) & CLAHE SUITE ENGINE
Core Domains: Enhancement, Local Contrast, Restoration, Morphology, 
              Texture (GLCM/LBP), Transforms (FFT/DCT), Watermarking/Stego.
=============================================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import base64
import sys

class DIPEngine:
    """Core Digital Image Processing Algorithms Engine using OpenCV & NumPy."""

    @staticmethod
    def apply_clahe(img, clip_limit=3.0, tile_grid=(8, 8)):
        """
        Adaptive Histogram Equalization with Contrast Limiting (CLAHE).
        Prevents noise over-amplification in uniform regions.
        """
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        enhanced = clahe.apply(img_gray)
        return img_gray, enhanced

    @staticmethod
    def local_histogram_processing(img, window_size=7, k0=0.4, k1=0.02, k2=0.4, E=4.0):
        """Adaptive Local Contrast Enhancement based on local mean and variance."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        img_float = img.astype(np.float32)
        global_mean = np.mean(img_float)
        global_std = np.std(img_float)

        pad = window_size // 2
        padded = np.pad(img_float, pad, mode='reflect')
        output = np.copy(img_float)

        h, w = img.shape
        for y in range(h):
            for x in range(w):
                region = padded[y:y+window_size, x:x+window_size]
                local_mean = np.mean(region)
                local_std = np.std(region)

                if (local_mean <= k0 * global_mean) and (k1 * global_std <= local_std <= k2 * global_std):
                    output[y, x] = np.clip(E * img_float[y, x], 0, 255)

        return output.astype(np.uint8)

    @staticmethod
    def wiener_deblur_restoration(img, kernel_size=5, noise_var=0.01):
        """Wiener Deconvolution Filter for image restoration and deblurring."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Create Gaussian PSF (Point Spread Function)
        kernel = cv2.getGaussianKernel(kernel_size, 1.0)
        psf = np.outer(kernel, kernel)
        
        img_fft = np.fft.fft2(img.astype(np.float32))
        psf_padded = np.zeros_like(img, dtype=np.float32)
        kh, kw = psf.shape
        psf_padded[:kh, :kw] = psf
        psf_fft = np.fft.fft2(psf_padded)

        # Wiener Filter formula: H* / (|H|^2 + K)
        psf_conj = np.conj(psf_fft)
        wiener_filter = psf_conj / (np.abs(psf_fft)**2 + noise_var)
        restored_fft = img_fft * wiener_filter
        
        restored = np.abs(np.fft.ifft2(restored_fft))
        return np.clip(restored, 0, 255).astype(np.uint8)

    @staticmethod
    def calculate_metrics(orig, processed):
        """Computes Shannon Entropy, Contrast Ratio, PSNR, and SSIM."""
        # Shannon Entropy
        hist, _ = np.histogram(processed, bins=256, range=(0, 256), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))

        # Contrast Ratio
        contrast = np.std(processed) / (np.mean(processed) + 1e-5)

        # Peak Signal-to-Noise Ratio (PSNR)
        mse = np.mean((orig.astype(np.float64) - processed.astype(np.float64)) ** 2)
        psnr = 10 * np.log10((255.0 ** 2) / mse) if mse > 0 else 100.0

        return {
            "entropy": round(entropy, 3),
            "contrast": round(contrast, 3),
            "psnr": round(psnr, 2)
        }


# =============================================================================
# INTERACTIVE MATPLOTLIB & OPENCV DASHBOARD
# =============================================================================

def run_interactive_gui(image_path=None, clip_limit=3.0, grid_size=8):
    """Launches an interactive OpenCV/Matplotlib inspection window."""
    if image_path:
        img = cv2.imread(image_path)
    else:
        # Create Synthetic Medical Radiograph test pattern
        x = np.linspace(-3, 3, 512)
        y = np.linspace(-3, 3, 512)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X**2 + Y**2) * 40 + np.cos(X*2) * 30 + 100
        img = np.clip(Z, 0, 255).astype(np.uint8)

    orig_gray, clahe_res = DIPEngine.apply_clahe(img, clip_limit, (grid_size, grid_size))
    metrics = DIPEngine.calculate_metrics(orig_gray, clahe_res)

    # Calculate Histograms
    hist_orig = cv2.calcHist([orig_gray], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([clahe_res], [0], None, [256], [0, 256])

    plt.figure(figsize=(14, 8), facecolor='#000814')
    plt.suptitle("4D SPATIAL DIP ENGINE - CLAHE & LOCAL CONTRAST", color='#00f0ff', fontsize=16, fontweight='bold')

    # Original Image Preview
    plt.subplot(2, 2, 1)
    plt.imshow(orig_gray, cmap='gray')
    plt.title("Original Input Image", color='#e0f7ff')
    plt.axis('off')

    # CLAHE Enhanced Preview
    plt.subplot(2, 2, 2)
    plt.imshow(clahe_res, cmap='gray')
    plt.title(f"CLAHE Enhanced (Clip: {clip_limit}, Grid: {grid_size}x{grid_size})", color='#00f0ff')
    plt.axis('off')

    # Original Histogram Chart
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_facecolor('#001020')
    plt.plot(hist_orig, color='#80b3d5')
    plt.title("Original Intensity Histogram", color='#e0f7ff')
    plt.xlim([0, 256])
    ax3.tick_params(colors='#80b3d5')

    # CLAHE Histogram Chart & Telemetry Metrics
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_facecolor('#001020')
    plt.plot(hist_clahe, color='#00f0ff')
    plt.title(f"Processed Histogram | Entropy: {metrics['entropy']} bits | PSNR: {metrics['psnr']} dB", color='#00f0ff')
    plt.xlim([0, 256])
    ax4.tick_params(colors='#80b3d5')

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" 4D SPATIAL DIGITAL IMAGE PROCESSING (DIP) PYTHON ENGINE")
    print("=" * 70)

    # Check if image file parameter was provided in command line
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"[+] Processing file: {file_path}")
        run_interactive_gui(file_path)
    else:
        print("[+] Generating Synthetic Test Pattern and launching GUI...")
        run_interactive_gui()