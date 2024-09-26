import os

import PIL
import numpy as np
from PIL import Image
import cv2
import utils


""" calc_pixels_qr_code(img, sought)
        - image: image path for the image that contains the qrcode detected, contoured and filled
        - sought: contour color to use in searching
"""
def calc_pixels_qr_code(img, sought):
    # Open image and make into numpy array
    im = np.array(Image.open(img).convert('RGB'))

    # Find all pixels where the 3 RGB values match "sought", and count them
    result = np.count_nonzero(np.all(im == sought, axis=2))

    return result


""" calc_pixels_qr_code_width(img, sought)
        - image: image path for the image that contains the qrcode detected, contoured and filled
        - sought: contour color to use in searching
"""
def calc_pixels_qr_code_width(img, sought):

    # array that contains the pixel count for each line
    count_per_line = []
    color_array = np.array(sought)
    # Open image and make into numpy array
    im = np.array(Image.open(img).convert('RGB'))

    # Get the shape of the array (height, width, channels for RGB image)
    height, width, channels = im.shape

    # Traverse the image line by line
    for y in range(height):
        count = 0
        for x in range(width):
            # get pixel data
            pixel_value = im[y, x]
            # check if the color of the pixel is equal to the sought
            if (np.array_equal(pixel_value, color_array)):
                count += 1
        count_per_line.append(count)

    # filter the array that contains the pixel count for each line, deleting all the positions pixel count = 0
    clean_count_list = [i for i in count_per_line if i != 0]

    # calculate the average pixel count of the lines
    average_line = round(sum(clean_count_list) / len(clean_count_list), 0)

    return average_line

""" calc_pixels_qr_code_height(img, sought)
        - image: image path for the image that contains the qrcode detected, contoured and filled
        - sought: contour color to use in searching
"""
def calc_pixels_qr_code_height(img, sought):

    # array that contains the pixel count for each line
    count_per_col = []
    color_array = np.array(sought)
    # Open image and make into numpy array
    im = np.array(Image.open(img).convert('RGB'))

    # Get the shape of the array (height, width, channels for RGB image)
    height, width, channels = im.shape

    # Traverse the image col by col
    for x in range(width):
        count = 0
        for y in range(height):
            # get pixel data
            pixel_value = im[y, x]
            # check if the color of the pixel is equal to the sought
            if (np.array_equal(pixel_value, color_array)):
                count += 1
        count_per_col.append(count)

    # filter the array that contains the pixel count for each column, deleting all the positions pixel count = 0
    clean_count_list = [i for i in count_per_col if i != 0]

    # calculate the average pixel count of the columns
    average_col = round(sum(clean_count_list) / len(clean_count_list), 0)

    # print(average_col)
    return average_col




def detect_white_gray(img):
    img = "resultados_rede/" + img
    image = np.array(Image.open(img).convert('RGB'))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    invGamma = 1.0 / 0.7
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    gray = cv2.LUT(gray, table)
    cv2.imwrite("gray/image_gray.jpg", gray)

    image = np.array(Image.open("gray/image_gray.jpg").convert('RGB'))

    color = np.array([0, 0, 0])
    L2 = np.sqrt(np.sum((image - color) ** 2, axis=2))  # L2 distance of each pixel from color

    img_dim = image.shape
    new_img = np.zeros((img_dim[0], img_dim[1]))
    new_img[L2 < 20] = 255

    # to change color of the detection
    """new_img = np.zeros((img_dim[0], img_dim[1], 3), dtype=np.uint8)  # Adicionando o terceiro canal para cores RGB
    new_img[L2 > 20] = [255, 0, 0]
    new_img[L2 < 20] = [255, 255, 255]"""

    im = Image.fromarray(new_img)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("resultados_dimensoes/result_image_dimensions.jpg")



def calc_pixels_window(image, window, degrees, img_number):
    """wxx = np.array(pontos_janela_x).reshape(-1, 1)
                    wyy = np.array(pontos_janela_y).reshape(-1, 1)
                    plt.scatter(wxx, wyy, color='g')

                    plt.show()"""
    (x, y, w, h) = window
    top_left = (x, y)
    bottom_x = x+w
    bottom_y = y+h

    height, width, channels = image.shape

    if bottom_x >= width:
        bottom_x = width-1
    if bottom_y >= height:
        bottom_y = height-1

    bottom_right = (bottom_x, bottom_y)

    image_window_block = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    image_window_block_np_array = np.array(image_window_block)

    # rotate windowed_image by the slope angle calculated before

    image_window_block_converted = Image.fromarray(image_window_block_np_array)

    if image_window_block_converted.mode != 'RGB':
        image_window_block_converted = image_window_block_converted.convert('RGB')

    image_name = "window_rotated/BEFORE_ROTATED_" + str(img_number) + ".jpg"
    image_window_block_converted.save(image_name)
    black = (0, 0, 0)

    #-----------------------

    rotated_image = image_window_block_converted

    if rotated_image.mode != 'RGB':
        rotated_image = rotated_image.convert('RGB')

    rotated_width, rotated_height = rotated_image.size

    rotated_image = image_window_block_converted.rotate(degrees, PIL.Image.NEAREST, expand=True, fillcolor=black)

    if rotated_image.mode != 'RGB':
        rotated_image = rotated_image.convert('RGB')
    rotated_image.save("window_rotated/ROTATED_" + str(img_number) + ".jpg")

    rotated_width, rotated_height = rotated_image.size
    # Calculo das larguras da janela depois da regressão linear e rotação segundo o declive

    window_average_line_pixel_count = 0
    valid_lines = 0
    for i in range(rotated_height):
        line_res = calc_pixels_width_by_line_comparison_not_equal_sought(rotated_image, rotated_width, black, i)
        if line_res != -1:
            window_average_line_pixel_count += line_res
            valid_lines += 1

    if valid_lines != 0:
        window_average_line_pixel_count = round(window_average_line_pixel_count/valid_lines, 0)
    else:
        window_average_line_pixel_count = None

    return window_average_line_pixel_count


def calc_pixels_width_by_line_comparison_not_equal_sought(window, width, sought, line):  # sought = cor do contorno
    count_per_line = []
    color_array = np.array(sought)

    image_np = np.array(window)
    # Traverse the image line by line
    pixel_anterior = image_np[0, 0]
    start_count = False

    count = 0
    for x in range(width):

        pixel_value = image_np[line, x]

        if not(np.array_equal(pixel_value, color_array)):
            if not np.array_equal(pixel_value, pixel_anterior) and start_count == True:
                start_count = False
                count_per_line.append(count)
        else:

            count += 1
            start_count = True

        pixel_anterior = image_np[line, x]

    if len(count_per_line) != 0:
        average_line = round(sum(count_per_line) / len(count_per_line), 0)
    else:
        average_line = -1

    return average_line


def calc_pixels_width(img, sought):  # sought = cor do contorno

    count_per_line = []
    color_array = np.array(sought)
    # Open image and make into numpy array
    im = np.array(Image.open(img).convert('RGB'))

    # Get the shape of the array (height, width, channels for RGB image)
    height, width, channels = im.shape

    # Traverse the image line by line
    pixel_anterior = im[0, 0]
    start_count = False
    for y in range(height):
        count = 0
        for x in range(width):
            pixel_value = im[y, x]
            if (np.array_equal(pixel_value, color_array)):
                count += 1
                start_count = True
            else:
                if not np.array_equal(pixel_value, pixel_anterior) and start_count == True:
                    start_count = False
                    count_per_line.append(count)

            pixel_anterior = im[y, x]

    average_line = round(sum(count_per_line) / len(count_per_line), 0)

    # print(average_line)
    return average_line


def analyse_all_images():
    origin_folder_path = 'origin'

    file_names = os.listdir(origin_folder_path)


    for i in range(len(file_names)):
        try:
            origin_image = file_names[i]

            detect_white_gray(origin_image)
            im = np.array(Image.open("resultados_dimensoes/result_image_dimensions.jpg").convert('RGB'))

            qr_code_side = int(origin_image[0])

            qr_area, qr_area_real, qr_width = utils.calc_pixels_e_area_qrcode(origin_image,
                                                                              comp_real_qr=qr_code_side,
                                                                              largura_real_qr=qr_code_side)

            val = calc_pixels_width("resultados_dimensoes/result_image_dimensions.jpg", [0, 0, 0])

            # pixel correlation
            real_window_average_width = (val * qr_code_side) / qr_width

            # convert to mm
            real_window_average_width = real_window_average_width * 10

            print("\tDimensões Fenda: " + str(real_window_average_width) + " mm")


        except:
            continue


if __name__ == '__main__':
    analyse_all_images()
