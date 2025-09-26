# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:
    
    def translate_image(image, shift_x=50, shift_y=30):
        h, w = image.shape
        translated = np.zeros_like(image)
        
        # calcula os limites válidos para evitar índices fora dos limites
        start_y = max(0, shift_y)
        end_y = min(h, h)
        start_x = max(0, shift_x)
        end_x = min(w, w)
        
        if shift_y >= 0 and shift_x >= 0:
            translated[shift_y:, shift_x:] = image[:-shift_y if shift_y > 0 else h, 
                                                 :-shift_x if shift_x > 0 else w]
        
        return translated
    

    def rotate_image_90_clockwise(image):
        return np.rot90(image, k=-1)  # k=-1 para rotação horária
    
    
    def stretch_image_horizontal(image, scale_factor=1.5):
        h, w = image.shape
        new_w = int(w * scale_factor)
        stretched = np.zeros((h, new_w))
        
        # nearest neighbor
        for i in range(h):
            for j in range(new_w):
                # mapeia a coordenada nova para a coordenada original
                orig_j = int(j / scale_factor)
                if orig_j < w:
                    stretched[i, j] = image[i, orig_j]
        
        return stretched
    
    
    def mirror_image_horizontal(image):
        """Espelha a imagem horizontalmente (flip vertical axis)"""
        return np.fliplr(image)
    

    def barrel_distortion(image):
        """Aplica uma distorção barrel simples usando função radial"""
        h, w = image.shape
        distorted = np.zeros_like(image)
        
        center_x, center_y = w // 2, h // 2
        
        k = 0.0002  # dá para ajudar este valor para controlar a intensidade
        
        for y in range(h):
            for x in range(w):
                dx = x - center_x
                dy = y - center_y
                r = np.sqrt(dx**2 + dy**2)
                
                # aplica a distorção radial
                r_distorted = r * (1 + k * r**2)
                
                if r > 0:  # evita divisão por zero
                    scale = r_distorted / r
                    new_x = center_x + dx * scale
                    new_y = center_y + dy * scale
                    
                    if (0 <= new_x < w-1) and (0 <= new_y < h-1):
                        # interpolação bilinear simples
                        x1, y1 = int(new_x), int(new_y)
                        x2, y2 = x1 + 1, y1 + 1
                        
                        wx = new_x - x1
                        wy = new_y - y1
                        
                        # interpolação bilinear
                        val = (image[y1, x1] * (1-wx) * (1-wy) +
                               image[y1, x2] * wx * (1-wy) +
                               image[y2, x1] * (1-wx) * wy +
                               image[y2, x2] * wx * wy)
                        
                        distorted[y, x] = val
                    else:
                        distorted[y, x] = 0  # pixels fora dos limites ficam pretos
                else:
                    distorted[y, x] = image[y, x]  # centro não é distorcido
        
        return distorted
    
    results = {
        "translated": translate_image(img),
        "rotated": rotate_image_90_clockwise(img),
        "stretched": stretch_image_horizontal(img),
        "mirrored": mirror_image_horizontal(img),
        "distorted": barrel_distortion(img)
    }
    
    return results

# exemplo
if __name__ == "__main__":
    test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    transformations = apply_geometric_transformations(test_image)
    
    print("\nTransformações aplicadas:")
    for name, transformed_img in transformations.items():
        print(f"{name}: shape = {transformed_img.shape}, dtype = {transformed_img.dtype}")
        print(f"  Min value: {transformed_img.min()}, Max value: {transformed_img.max()}\n")
    
    print("\nFunção implementada com sucesso!")
