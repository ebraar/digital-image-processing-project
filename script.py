import cv2 # type: ignore
import numpy as np # type: ignore
from scipy.signal import wiener # type: ignore

# 2D konvolüsyon işlemini yapan fonksiyon
def apply_2d_convolution(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    
    # Resmi padding ile genişletme
    padded_image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    # Çıkış resmi
    output = np.zeros_like(image, dtype=np.float32)
    
    # Konvolüsyon işlemi
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)

# 1. Resmi okuma
image = cv2.imread('foto.png')
if image is None:
    print("Resim yüklenemedi!")
else:
    # 2. Gri tonluya çevirme
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray_image.png', gray_image)

    # 3. Gauss gürültüsü ekleme
    mean = 0
    std_dev = 25
    gauss_noise = np.random.normal(mean, std_dev, gray_image.shape).astype(np.float32)
    noisy_image = gray_image + gauss_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    cv2.imwrite('noisy_image.png', noisy_image)

    # 4. Gürültüyü temizleme
    # Mean filtresi
    mean_filter = cv2.blur(noisy_image, (5, 5))
    cv2.imwrite('mean_filter_result.png', mean_filter)

    # Wiener filtresi
    wiener_filter = wiener(noisy_image, (5, 5))
    wiener_filter = np.clip(wiener_filter, 0, 255).astype(np.uint8)
    cv2.imwrite('wiener_filter_result.png', wiener_filter)

    # 5. OTSU yöntemi ile siyah-beyaz dönüştürme
    _, otsu_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('otsu_image.png', otsu_image)

    # 6. Morfolojik açma işlemi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_open = cv2.morphologyEx(otsu_image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('morph_open_result.png', morph_open)

    # 7. 2D Konvolüsyon İşlemleri
    # Mean filtresi (Gürültü giderme)
    mean_kernel = np.ones((3, 3), dtype=np.float32) / 9
    mean_conv_result = apply_2d_convolution(gray_image, mean_kernel)
    cv2.imwrite('conv_mean_result.png', mean_conv_result)

    # Keskinleştirme filtresi
    sharpen_kernel = np.array([[0, -1, 0], 
                                [-1, 5, -1], 
                                [0, -1, 0]], dtype=np.float32)
    sharpen_conv_result = apply_2d_convolution(gray_image, sharpen_kernel)
    cv2.imwrite('conv_sharpen_result.png', sharpen_conv_result)

    # Kenar bulma filtresi
    edge_kernel = np.array([[-1, -1, -1], 
                            [-1, 8, -1], 
                            [-1, -1, -1]], dtype=np.float32)
    edge_conv_result = apply_2d_convolution(gray_image, edge_kernel)
    cv2.imwrite('conv_edge_result.png', edge_conv_result)

    # Sonuçları görüntüleme
    cv2.imshow("Orijinal Resim", image)  # Orijinal Renkli Resim
    cv2.imshow("Gri Tonlu Resim", gray_image)
    cv2.imshow("Gauss Gurultulu Resim", noisy_image)
    cv2.imshow("Mean Filtresi Sonucu", mean_filter)
    cv2.imshow("Wiener Filtresi Sonucu", wiener_filter)
    cv2.imshow("OTSU Siyah Beyaz Resim", otsu_image)
    cv2.imshow("Morfolojik Acma Sonucu", morph_open)
    cv2.imshow("Mean Konvolusyon Sonucu", mean_conv_result)
    cv2.imshow("Keskinlestirme Konvolusyon Sonucu", sharpen_conv_result)
    cv2.imshow("Kenar Bulma Konvolusyon Sonucu", edge_conv_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()