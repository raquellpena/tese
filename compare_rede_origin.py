import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def calc_pixels_detection(img, sought):
    # Open image and make into numpy array
    im = np.array(Image.open(img).convert('RGB'))

    # Find all pixels where the 3 RGB values match "sought", and count them
    result = np.count_nonzero(np.all(im != sought, axis=2))

    return result
def compare_NN_detection_precision():
    origin_folder_path = 'origin_marcada'
    rede_folder_path = 'rede_marcada'

    file_names = os.listdir(origin_folder_path)

    sufix = '_rede.jpg'

    prefixed_file_names = [file[:-4] + sufix for file in file_names]

    for i in range(len(file_names)):
        try:
            origin_image = origin_folder_path + '/' + file_names[i]
            NN_image = rede_folder_path + '/' + prefixed_file_names[i]

            black = [0, 0, 0]

            pixel_count_origin = calc_pixels_detection(origin_image, black)

            pixel_count_NN = calc_pixels_detection(NN_image, black)

            error = (pixel_count_NN * 100)/pixel_count_origin

            print(file_names[i])
            print("Pixel count origin: " + str(pixel_count_origin))
            print("Pixel count NN: " + str(pixel_count_NN))
            print(str(error) + "%\n---------------------------")
        except:
            continue


# Function to calculate the Dice coefficient
def dice_coefficient(img1, img2):
    # Ensure both images have the same size
    assert img1.shape == img2.shape, "Images must be the same shape"

    # Flatten the images to 1D arrays
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # Calculate the intersection between the two images
    intersection = np.sum(img1_flat * img2_flat)

    # Calculate the Dice coefficient
    dice = (2 * intersection) / (np.sum(img1_flat) + np.sum(img2_flat))

    return dice

def compare_NN_detection_precision_DICE():
    origin_folder_path = 'origin_marcada'
    rede_folder_path = 'resultados_rede'

    file_names = os.listdir(origin_folder_path)

    for i in range(len(file_names)):
        try:
            origin_image = origin_folder_path + '/' + file_names[i]
            NN_image = rede_folder_path + '/' + file_names[i]

            # Load the two images in color
            img1 = cv2.imread(origin_image)
            img2 = cv2.imread(NN_image)

            # Convert images to grayscale
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Aplica o threshold para converter em preto e branco (binário)
            _, img2_bin = cv2.threshold(img2_gray, 10, 255, cv2.THRESH_BINARY)
            img1_bin = img1_gray

            # # Exibe as imagens binarizadas
            # plt.figure(figsize=(10, 5))
            #
            # # Exibindo a primeira imagem
            # plt.subplot(1, 2, 1)
            # plt.imshow(img1_bin, cmap='gray')
            # plt.title('Imagem 1 - Preto e Branco')
            #
            # # Exibindo a segunda imagem
            # plt.subplot(1, 2, 2)
            # plt.imshow(img2_bin, cmap='gray')
            # plt.title('Imagem 2 - Preto e Branco')
            #
            # plt.show()

            # Calculate the Dice coefficient
            dice_score = dice_coefficient(img1_bin, img2_bin)
            print(file_names[i])
            print(f"Dice coefficient: {dice_score:.4f}\n-----------------------------------------")

        except:
            continue



if __name__ == '__main__':
    print("------------------ NOSSO -----------------")
    compare_NN_detection_precision()
    print("------------------ DICE -----------------")
    compare_NN_detection_precision_DICE()
