import cv2
import numpy as np
from scipy.signal import wiener

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

    # 3. Resmi 256x256 boyutlarına yeniden boyutlandırma
    resized_image = cv2.resize(gray_image, (256, 256))
    cv2.imwrite('resized_image.png', resized_image)

    # 4. Gauss gürültüsü ekleme
    mean = 0
    std_dev = 25
    gauss_noise = np.random.normal(mean, std_dev, resized_image.shape).astype(np.float32)
    noisy_image = resized_image + gauss_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    cv2.imwrite('noisy_image.png', noisy_image)

    # 5. Gürültüyü temizleme
    # Mean filtresi
    mean_filter = cv2.blur(noisy_image, (5, 5))
    cv2.imwrite('mean_filter_result.png', mean_filter)

    # Wiener filtresi
    wiener_filter = wiener(noisy_image, (5, 5))
    wiener_filter = np.clip(wiener_filter, 0, 255).astype(np.uint8)
    cv2.imwrite('wiener_filter_result.png', wiener_filter)

    # 6. Sobel kenar bulma operatörü ile kenar bulma
    sobel_x = cv2.Sobel(wiener_filter, cv2.CV_64F, 1, 0, ksize=3)  # X yönü
    sobel_y = cv2.Sobel(wiener_filter, cv2.CV_64F, 0, 1, ksize=3)  # Y yönü

    # Mutlak değer ve 8-bit formatına dönüştürme
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # X ve Y yönündeki kenarları birleştirme
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    cv2.imwrite('sobel_combined_result.png', sobel_combined)

    # 7. OTSU yöntemi ile siyah-beyaz dönüştürme
    _, otsu_image = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite('otsu_image.png', otsu_image)

    # 8. Morfolojik açma işlemi
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_open = cv2.morphologyEx(otsu_image, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('morph_open_result.png', morph_open)

    # 9. 2D Konvolüsyon İşlemleri
    # Mean filtresi (Gürültü giderme)
    mean_kernel = np.ones((3, 3), dtype=np.float32) / 9
    mean_conv_result = apply_2d_convolution(resized_image, mean_kernel)
    cv2.imwrite('conv_mean_result.png', mean_conv_result)

    # Keskinleştirme filtresi
    sharpen_kernel = np.array([[0, -1, 0], 
                                [-1, 5, -1], 
                                [0, -1, 0]], dtype=np.float32)
    sharpen_conv_result = apply_2d_convolution(resized_image, sharpen_kernel)
    cv2.imwrite('conv_sharpen_result.png', sharpen_conv_result)

    # Kenar bulma filtresi
    edge_kernel = np.array([[-1, -1, -1], 
                            [-1, 8, -1], 
                            [-1, -1, -1]], dtype=np.float32)
    edge_conv_result = apply_2d_convolution(resized_image, edge_kernel)
    cv2.imwrite('conv_edge_result.png', edge_conv_result)

    # Sonuçları görüntüleme
    cv2.imshow("Orijinal Resim", image)  # Orijinal Renkli Resim
    cv2.imshow("Gri Tonlu Resim", gray_image)
    cv2.imshow("256x256 Boyutunda Resim", resized_image)
    cv2.imshow("Gauss Gurultulu Resim", noisy_image)
    cv2.imshow("Mean Filtresi Sonucu", mean_filter)
    cv2.imshow("Wiener Filtresi Sonucu", wiener_filter)
    cv2.imshow("Sobel Kenar Bulma Sonucu", sobel_combined)
    cv2.imshow("OTSU Siyah Beyaz Resim", otsu_image)
    cv2.imshow("Morfolojik Acma Sonucu", morph_open)
    cv2.imshow("Mean Konvolusyon Sonucu", mean_conv_result)
    cv2.imshow("Keskinlestirme Konvolusyon Sonucu", sharpen_conv_result)
    cv2.imshow("Kenar Bulma Konvolusyon Sonucu", edge_conv_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()