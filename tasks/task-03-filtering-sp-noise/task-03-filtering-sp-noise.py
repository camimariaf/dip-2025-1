import cv2 # estava só cv, então mudei para cv2 para rodar o código
import numpy as np

def remove_salt_and_pepper_noise(image: np.ndarray) -> np.ndarray:
    """
    Removes salt and pepper noise from a grayscale image.

    Parameters:
        image (np.ndarray): Noisy input image (grayscale).

    Returns:
        np.ndarray: Denoised image.
    """
    denoised = cv2.medianBlur(image, 5)
    return denoised

if __name__ == "__main__":
    noisy_image = cv2.imread("head.png", cv2.IMREAD_GRAYSCALE)

    if noisy_image is None:
        print("Erro: Não foi possível carregar a imagem 'head.png'")
        exit(1)
        
    denoised_image = remove_salt_and_pepper_noise(noisy_image)
    cv2.imwrite("head_filtered.png", denoised_image)
