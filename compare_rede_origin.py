import os
import numpy as np
from PIL import Image

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



if __name__ == '__main__':
    compare_NN_detection_precision()
