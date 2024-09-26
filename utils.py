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


    print("--- Image:", image, "\n\tQRCode Nº Pixeis QRArea: ", qr_area, "\n\tQR Code Cm Reais: ", qr_area_real)

    return qr_area, qr_area_real, qr_width



