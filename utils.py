import pixelcount
import find_paper

""" func_m(image,type)
    - image: image name
    - type: will be used for selecting type of crack(longitudinal,crocodile, ...)
"""
def calc_pixels_e_area_qrcode(image,comp_real_qr, largura_real_qr):

    # QRCode - define real qrcode width and height in cm and calculate the area of the qrcode
    qr_area_real = largura_real_qr*comp_real_qr

    # call the function that detects the qrcode, contour and fill it and saves the image in a file
    find_paper.find_qrcode(image)

    # define the path for the image created above
    con = './contours/contoured_' + image

    # calculate the average pixel width of the qrcode in the image defined above, searching by pixel info
    qr_width = pixelcount.calc_pixels_qr_code_width(con, [254, 0, 0])

    # calculate the average pixel height of the qrcode in the image defined above, searching by pixel info - NOT USED - only used if necessary to calculate vertically, this means, by height - average pixel height - in case the crack used next is in horizontal orientation
    qr_height = pixelcount.calc_pixels_qr_code_height(con, [254, 0, 0])


    # calculate the crack pixel area
    qr_area = pixelcount.calc_pixels_qr_code(con,[254, 0, 0])
    '''
    # get image name by splitting by "." so it removes the extension
    image_name = image.split(".")[0]

    # add the new extension to the image name
    image_name = image_name + ".jpg"

    # call the function that detects and creates the image with the crack in black/white
    pixelcount.detect_white_gray(image_name)

    # calculate the average pixel width of the crack, searching by pixel info
    average_line = pixelcount.calc_pixels_width("resultados_dimensoes/result_image_dimensions.jpg", [0, 0, 0])

    # calculates the total number of pixels of the crack for the area calculation, searching by pixel info
    total_pixels_crack = pixelcount.calc_pixels_crack("resultados_dimensoes/result_image_dimensions.jpg", [0, 0, 0])

    # calculating the correlation for the real width of the crack
    average_width = (average_line * m_l)/qr_width

    # calculating the correlation for the real area of the crack
    crack_area = (total_pixels_crack * m_area)/qr_area'''

    print("--- Nº Pixeis QRArea \t: ", qr_area, "\n\tCm: ", qr_area_real)
    #print("--- Nº Pixeis \t: ", average_line, "\n\tCm: ", average_width)
    #print("--- Nº Pixeis Area \t: ", total_pixels_crack, "\n\tCm2: ", crack_area)

    # calculating pixel width and the correlation for the real width of the crack in line 100, just for debugging
    #with_100 = pixelcount.calc_pixels_width_by_line("resultados_dimensoes/result_image_dimensions.jpg", [0, 0, 0],100)
    #with_100_cm = (with_100 * m_l)/qr_width
    #print("--- Nº Pixeis na linha 100 \t: ", with_100, "\n\tCm: ", with_100_cm)
    return qr_area, qr_area_real

# run func
calc_pixels_e_area_qrcode("1.jpg", 5,5)


