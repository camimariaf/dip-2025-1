# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import scikitimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:

    matched_img = np.zeros_like(source_img, dtype=np.uint8)
    
    for channel in range(3):  # 0=red, 1=green, 2=blue
        # extrai o canal atual de ambas as imagens
        source_channel = source_img[:, :, channel]
        reference_channel = reference_img[:, :, channel]
        
        matched_channel = exposure.match_histograms(source_channel, reference_channel)
        
        # assegura que os valores estão no intervalo correto e tipo uint8
        matched_img[:, :, channel] = matched_channel.astype(np.uint8)
    
    return matched_img


# implementação alternativa usando correspondência manual de histograma
def match_histograms_rgb_manual(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:

    def match_histogram_channel(source_channel, reference_channel):
        source_hist, _ = np.histogram(source_channel.flatten(), bins=256, range=(0, 256))
        reference_hist, _ = np.histogram(reference_channel.flatten(), bins=256, range=(0, 256))
        
        # calcula funções de distribuição cumulativas
        source_cdf = np.cumsum(source_hist).astype(np.float64)
        reference_cdf = np.cumsum(reference_hist).astype(np.float64)
        
        # normaliza CDFs to [0, 1]
        source_cdf = source_cdf / source_cdf[-1]
        reference_cdf = reference_cdf / reference_cdf[-1]
        
        lookup_table = np.zeros(256, dtype=np.uint8)
        
        for i in range(256):
            diff = np.abs(reference_cdf - source_cdf[i])
            lookup_table[i] = np.argmin(diff)
            
        # aplica a tabela de consulta para mapear os valores do canal de origem
        matched_channel = lookup_table[source_channel]
        
        return matched_channel
    
    matched_img = np.zeros_like(source_img, dtype=np.uint8)
    
    # processa cada canal RGB
    for channel in range(3):
        source_channel = source_img[:, :, channel]
        reference_channel = reference_img[:, :, channel]
        
        matched_channel = match_histogram_channel(source_channel, reference_channel)
        matched_img[:, :, channel] = matched_channel
    
    return matched_img


# exemplo
if __name__ == "__main__":
    source_img = cv.imread('source.jpg')
    reference_img = cv.imread('reference.jpg')
    
    source_img = cv.cvtColor(source_img, cv.COLOR_BGR2RGB)
    reference_img = cv.cvtColor(reference_img, cv.COLOR_BGR2RGB)
    
    matched_img = match_histograms_rgb(source_img, reference_img)
    
    matched_img_bgr = cv.cvtColor(matched_img, cv.COLOR_RGB2BGR)
    
    cv.imwrite('output.jpg', matched_img_bgr)
    
    print("Concluído com sucesso!")
