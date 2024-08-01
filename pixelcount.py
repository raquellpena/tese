import PIL
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import utils
from sklearn.linear_model import LinearRegression


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


def detect_white_gray_crocodile(img):
    image = np.array(Image.open(img).convert('RGB'))
    color = np.array([0, 0, 0])
    L2 = np.sqrt(np.sum((image - color) ** 1.8, axis=2))  # L2 distance of each pixel from color

    img_dim = image.shape
    new_img = np.zeros((img_dim[0], img_dim[1]))
    new_img[L2 < 20] = 25

    im = Image.fromarray(new_img)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("resultados_dimensoes/result_image_dimensions.jpg")

    plt.subplot(121), plt.imshow(image)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_img, cmap='gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    plt.show()

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

    plt.show()
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(new_img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(121), plt.imshow(image)
    plt.show()

def get_slope_for_window(im, sought):
    # intermedio antes de fazer a função fixed_window_crack_width_calculation
    x_array = []
    y_array = []
    color_array = np.array(sought)

    # Open image and make into numpy array
    # im = np.array(Image.open(img).convert('RGB')) --> do this when calling the function

    # Get the shape of the array (height, width, channels for RGB image)
    height, width, channels = im.shape

    for y in range(height):

        for x in range(width):
            pixel_value = im[y, x]
            if (np.array_equal(pixel_value, color_array)):
                x_array.append(x)
                y_array.append(y)

    x = np.array(x_array).reshape(-1, 1)
    y = np.array(y_array).reshape(-1, 1)
    lrm = LinearRegression()
    lrm.fit(x[:300], y[:300])
    declive = lrm.coef_
    plt.scatter(x, y, color='g')
    plt.plot(x, lrm.predict(x), color='k')

    plt.show()

    return declive


def variable_window_crack_width_calculation(img, sought):  # por retificar

    color_array = np.array(sought)

    # Get the shape of the array (height, width, channels for RGB image)
    image_height, image_width, channels = im.shape

    window_height = 250
    starting_window_x = None
    end_window_x = -1

    # Traverse the image line by line
    pixel_anterior = im[0, 0]
    start_count = False
    y = 0
    x = 0
    window_x = 0
    while (y < image_height):

        if (y + window_height) > image_height:
            window_height = image_height - y

        while (x < image_width):
            for window_line in range(window_height):  # percorrer todas as linhas da window
                window_x = x
                while (window_x < image_width):
                    pixel_value = im[y + window_line, window_x]
                    if (np.array_equal(pixel_value,
                                       color_array) and starting_window_x == None):  # para todas as lihas da window, se para uma dada linha o pixel == preto e o starting_window_x ainda estiver a -1, atribuimos o valor do x atual
                        if starting_window_x == None or (starting_window_x > window_x):
                            starting_window_x = window_x
                        start_count = True
                    elif (not np.array_equal(pixel_value, pixel_anterior)) and start_count == True:
                        if end_window_x < window_x:
                            end_window_x = window_x
                        start_count = False

                    pixel_anterior = im[y, window_x]
                    window_x += 1
            # aqui já temos o startx mais à esquerda e o endx mais à direita
            # plot e resto do processamento

            # tratar da janela
            pontos_janela_x = []
            pontos_janela_y = []
            for wy in range(y, y + window_height):
                for wx in range(starting_window_x, end_window_x):
                    pixel_value = im[wy, wx]
                    if (np.array_equal(pixel_value, color_array)):
                        pontos_janela_x.append(wx)
                        pontos_janela_y.append(wy)

            wxx = np.array(pontos_janela_x).reshape(-1, 1)
            wyy = np.array(pontos_janela_y).reshape(-1, 1)
            plt.scatter(wxx, wyy, color='g')

            plt.show()

            # incrementamos o x para começar onde a janela anterior terminou
            x = end_window_x

            # resetar variaveis para continuar a ver janelas na horizontal
            pixel_anterior = im[0, 0]
            start_count = False
            starting_window_x = None
            end_window_x = -1

        y = y + window_height

def fixed_window_crack_width_calculation(img, sought):

    all_windows_average_line_pixel_count = 0


    img_number = 1

    color_array = np.array(sought)

    # Get the shape of the array (height, width, channels for RGB image)
    image_height, image_width, channels = im.shape

    print("H: %d W: %d" % (image_height, image_width))

    standard_value = 200
    window_height = standard_value
    window_width = standard_value

    last_x_pos = 0
    last_y_pos = 0

    n_janelas_total = 0

    janelas_brancas = 0
    janelas_com_dados = 0


    while (last_y_pos < image_height):
        while (last_x_pos < image_width):

            if (last_x_pos + window_width) > image_width:
                window_width = image_width - last_x_pos

            pontos_janela_x = []
            pontos_janela_y = []

            for wy in range(last_y_pos, last_y_pos + window_height):
                for wx in range(last_x_pos, last_x_pos + window_width):
                    # print("wy: %d wx: %d" % (wy, wx))
                    pixel_value = im[wy, wx]
                    if (np.array_equal(pixel_value, color_array)):
                        pontos_janela_x.append(wx)
                        pontos_janela_y.append(wy)

            n_janelas_total += 1
            min_len = 20
            if ((pontos_janela_x == [] and pontos_janela_y == []) or (
                    len(pontos_janela_x) < min_len and len(pontos_janela_y) < min_len)):
                janelas_brancas += 1
            else:
                janelas_com_dados += 1

                """wxx = np.array(pontos_janela_x).reshape(-1, 1)
                wyy = np.array(pontos_janela_y).reshape(-1, 1)
                plt.scatter(wxx, wyy, color='g')

                plt.show()"""

                top_left = (last_y_pos, last_x_pos)
                bottom_right = (last_y_pos + window_height, last_x_pos + window_width)

                image_window_block = im[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

                image_window_block_np_array = np.array(image_window_block)

                x = np.array(pontos_janela_x).reshape(-1, 1)
                y = np.array(pontos_janela_y).reshape(-1, 1)
                lrm = LinearRegression()
                lrm.fit(x, y)
                declive = lrm.coef_[0][0]
                angle = math.atan(declive)
                angle_degrees = math.degrees(angle)
                #print("DECLIVE JANELA ", janelas_com_dados, ": ", angle_degrees)

                plt.scatter(x, y, color='g')
                plt.plot(x, lrm.predict(x), color='k')

                plt.savefig('./windows_regressions/picture' + str(janelas_com_dados) + '.png')

                plt.show(block=False)
                plt.close()

                # rotate windowed_image by the slope angle calculated before

                image_window_block_converted = Image.fromarray(image_window_block_np_array)

                if image_window_block_converted.mode != 'RGB':
                    image_window_block_converted = image_window_block_converted.convert('RGB')

                image_name = "window_rotated/BEFORE_ROTATED_" + str(img_number) + ".jpg"
                image_window_block_converted.save(image_name)

                white = (255, 255, 255)
                rotated_image = image_window_block_converted.rotate(angle_degrees, PIL.Image.NEAREST, expand=True, fillcolor=white)

                if rotated_image.mode != 'RGB':
                    rotated_image = rotated_image.convert('RGB')
                rotated_image.save("window_rotated/ROTATED_" + str(img_number) + ".jpg")

                # Calculo das larguras da janela depois da regressão linear e rotação segundo o declive

                average_line_pixel_count_of_window = calc_pixels_width(image_name, [0, 0, 0])

                all_windows_average_line_pixel_count += average_line_pixel_count_of_window


                img_number += 1

            last_x_pos += window_width

        last_y_pos += window_height
        last_x_pos = 0
        window_width = standard_value

        if (last_y_pos + window_height) > image_height:
            window_height = image_height - last_y_pos

    average_of_all_windows_average_pixel_count_per_line = all_windows_average_line_pixel_count/janelas_com_dados

    print("TOTAL JANELAS: ", n_janelas_total)
    print("JANELAS COM DADOS: ", janelas_com_dados)
    print("JANELAS BRANCAS: ", janelas_brancas)
    print("MEDIA DE LARGURA REGRESSÃO:", average_of_all_windows_average_pixel_count_per_line)

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

    image_window_block = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    image_window_block_np_array = np.array(image_window_block)

    plt.imshow(image_window_block_np_array, interpolation='nearest')
    plt.show()

    plt.imshow(image, interpolation='nearest')
    plt.show()


    # rotate windowed_image by the slope angle calculated before

    image_window_block_converted = Image.fromarray(image_window_block_np_array)

    if image_window_block_converted.mode != 'RGB':
        image_window_block_converted = image_window_block_converted.convert('RGB')

    image_name = "window_rotated/BEFORE_ROTATED_" + str(img_number) + ".jpg"
    image_window_block_converted.save(image_name)

    black = (0, 0, 0)

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


def calc_pixels_crack(img, sought):  # sought = cor do contorno
    # Open image and make into numpy array
    im = np.array(Image.open(img).convert('RGB'))

    # Work out what we are looking for
    # sought = [0,255,0]

    # Find all pixels where the 3 RGB values match "sought", and count them
    result = np.count_nonzero(np.all(im == sought, axis=2))
    # print(result)
    return result


def calc_pixels_width_by_line(img, sought, line):  # sought = cor do contorno
    count_per_line = []
    color_array = np.array(sought)
    # Open image and make into numpy array
    im = np.array(Image.open(img).convert('RGB'))

    # Get the shape of the array (height, width, channels for RGB image)
    height, width, channels = im.shape

    # Traverse the image line by line
    pixel_anterior = im[0, 0]
    start_count = False

    count = 0
    for x in range(width):
        pixel_value = im[line, x]
        if (np.array_equal(pixel_value, color_array)):
            count += 1
            start_count = True
        else:
            if not np.array_equal(pixel_value, pixel_anterior) and start_count == True:
                start_count = False
                count_per_line.append(count)

        pixel_anterior = im[line, x]

    line_width = round(sum(count_per_line) / len(count_per_line), 0)

    return line_width

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

    # print(count_per_line)

    average_line = round(sum(count_per_line) / len(count_per_line), 0)

    # print(average_line)
    return average_line


if __name__ == '__main__':
    detect_white_gray("9.jpg")
    im = np.array(Image.open("resultados_dimensoes/result_image_dimensions.jpg").convert('RGB'))

    #fixed_window_crack_width_calculation(im, [0, 0, 0])

    print("FUNC 2")
    comp_real_qr = 5
    largura_real_qr = 5
    qr_area, qr_area_real, qr_width = utils.calc_pixels_e_area_qrcode("9.jpeg", comp_real_qr=comp_real_qr,
                                                                      largura_real_qr=largura_real_qr)

    val = calc_pixels_width("resultados_dimensoes/result_image_dimensions.jpg", [0, 0, 0])

    # pixel correlation
    real_window_average_width = (val * largura_real_qr) / qr_width

    # convert to mm
    real_window_average_width = real_window_average_width * 10

    print(str(real_window_average_width) + " mm")