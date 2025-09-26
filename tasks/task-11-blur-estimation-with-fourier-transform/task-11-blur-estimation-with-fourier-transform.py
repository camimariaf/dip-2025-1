"""
task-11-blur-estimation-with-fourier-transform.py

>>> IMPORTANT <<<
Implement the function `frequency_blur_score` below.

Rules:
- Keep the function name and signature EXACTLY the same.
- Do NOT use any external network calls.
- You may ONLY use standard Python, NumPy, and OpenCV (cv2).
- Return a single float (higher = sharper OR lower = blurrier, but be consistent).

Tip (from the FFT blur-detection tutorial):
- Convert to grayscale
- 2D FFT -> shift DC to center
- Zero-out a centered square (low frequencies)
- Magnitude spectrum (e.g., log1p(abs(...)))
- Use the mean magnitude of the remaining spectrum as the score
"""

from typing import Union
import numpy as np
import cv2


def frequency_blur_score(
    image: Union[np.ndarray, "cv2.Mat"],
    center_size: int = 60
) -> float:
    """
    Compute a blur/sharpness score in the frequency domain.

    Parameters
    ----------
    image : np.ndarray
        Input image, grayscale or BGR. Any dtype accepted; will be converted to float32.
    center_size : int, default=60
        Side length of the central square (low-frequency) region to suppress.

    Returns
    -------
    float
        A scalar score. You should make it so that SHARPER images get a HIGHER score.
        (This will align with the grader's expectation.)
    """
    # ====== YOUR CODE STARTS HERE ======
        if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = np.float32(gray)

    # a FFT move a frequência DC (baixa frequência) para o canto superior esquerdo
    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)

    # desloca a frequência DC para o centro do espectro para melhor visualização e processamento.
    dft_shifted = np.fft.fftshift(dft)

    # primeiro, obtém as dimensões da imagem
    rows, cols = gray.shape
    # calcula as coordenadas para o centro
    crow, ccol = rows // 2, cols // 2
    
    half_size = min(center_size // 2, crow, ccol)
    
    dft_masked = dft_shifted.copy()
    
    # zera a região de baixa frequência no centro
    dft_masked[crow - half_size:crow + half_size, ccol - half_size:ccol + half_size] = 0

    # inverte o deslocamento e calcula o espectro de magnitude
    dft_un_shifted = np.fft.ifftshift(dft_masked)
    
    # calcula a magnitude do espectro
    magnitude_spectrum = cv2.magnitude(dft_un_shifted[:, :, 0], dft_un_shifted[:, :, 1])
    #  tutoria suegere usar log1p
    magnitude_spectrum = np.log1p(magnitude_spectrum)

    # a média é calculada apenas nas áreas de alta frequência (que não foram zeradas)
    score = np.mean(magnitude_spectrum)
    
    return float(score)

# exemplo
if __name__ == '__main__':
    try:
        sharp_image = cv2.imread('sharp_image.jpg')
        blurry_image = cv2.imread('blurry_image.jpg')
        
        if sharp_image is None or blurry_image is None:
            print("Erro: Não foi possível carregar as imagens de teste. Certifique-se de que os arquivos 'sharp_image.jpg' e 'blurry_image.jpg' existem.")
        else:
            sharp_score = frequency_blur_score(sharp_image)
            blurry_score = frequency_blur_score(blurry_image)
            
            print(f"Pontuação da imagem nítida: {sharp_score:.2f}")
            print(f"Pontuação da imagem desfocada: {blurry_score:.2f}")

            if sharp_score > blurry_score:
                print("Teste de nitidez bem-sucedido! O score da imagem nítida foi maior.")
            else:
                print("Teste falhou. O score da imagem nítida foi menor ou igual ao da desfocada.")

    except Exception as e:
        print(f"Ocorreu um erro no bloco de execução: {e}")
    # ====== YOUR CODE ENDS HERE ======
