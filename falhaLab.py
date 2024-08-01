#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:39:40 2024

@author: nmm

v - visualisar grelha de anotação ou regressões lineares na grelha de anotação
p - escolher pré-processamento de imagem.
f - regressão linear no retangulo de seleção.
click (sem drag) ou cursoras - reposiciona retangulo seleção
k/l ou drag rato bottom right retangulo seleção - altera tamanho retangulo seleção
h/j - itera retangulo seleção pela grelha anotação

q - sair


"""

SEL_THRESHOLD = 4
SMALL_WINDOW = 40

from typing import List, Tuple
import numpy as np
import cv2
from matplotlib import pyplot as plt
import pygame
from pygame.locals import QUIT  # , K_v, K_p
# import math
from sklearn.linear_model import LinearRegression
import utils
import pixelcount
import math

declives = []
r_windows = []


class R_WINDOW:
    def __init__(self, window, res, declive):
        self.window = window
        self.res = res
        self.declive = declive


"""import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Line Visualization")
"""


# Function to load a TTF font
def load_font(font_path, size):
    """
    Load a TrueType font from the specified path and size.

    Parameters:
    - font_path (str): The path to the TrueType font file.
    - size (int): The font size.

    Returns:
    - pygame.font.Font: A Pygame font object.
    """
    return pygame.font.Font(font_path, size)


"""# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw a line with a number near its midpoint
    start = (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
    end = (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
    number = random.randint(0, 9)
    draw_line_with_number(screen, start, end, number, font)

    pygame.display.flip()

# Quit Pygame
pygame.quit()
"""


def draw_thick_lineX(screen, color, start_pos, end_pos, thickness):
    ### NOT USED!!!
    # Determine the direction of the line
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]

    # Determine the length of the line
    length = max(abs(dx), abs(dy))

    # Calculate the step size for each pixel
    if length != 0:
        step_x = dx / length
        step_y = dy / length
    else:
        step_x, step_y = 0, 0

    # Draw the line segment by segment
    for i in range(int(length)):
        # Calculate the alpha value (transparency) based on the distance from the start
        alpha = int(255 * (1 - i / length))
        # Create a surface with the desired color and alpha
        surface = pygame.Surface((thickness, thickness), pygame.SRCALPHA)
        surface.fill((color[0], color[1], color[2], alpha))
        # Calculate the position of the line segment
        pos = (int(start_pos[0] + i * step_x - thickness // 2), int(start_pos[1] + i * step_y - thickness // 2))
        # Draw the line segment
        screen.blit(surface, pos)


class ImageProcessor:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        self.original_image = cv2.imread(image_path)
        self.iheight, self.iwidth = self.image.shape[:2]
        self.preprocessed_image = None
        self.valid_positions = []
        self.screen = None
        self.rectangle = pygame.rect.Rect(200, 200, 20, 20, )
        self.declives = []

    def preprocess_image(self, min_sz: int = SMALL_WINDOW) -> None:
        """
        Preprocess the image by applying filters and dividing it into small sizes.

        Parameters:
        - min_sz (int): The minimum size threshold for small images. Default is 10.
        """
        img = self.apply_preprocessing_filters(self.image)
        self.preprocessed_image = pygame.image.frombuffer(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.image.shape[1::-1],
                                                          "RGB")
        self.valid_positions = self.divide_into_small_sizes(img, min_sz)

    def divide_into_small_sizes(self, image: np.ndarray, min_sz: int) -> List[Tuple[int, int, int, int]]:
        """
        Divide the image into small sizes based on the minimum size threshold.

        Parameters:
        - image (np.ndarray): The image to be divided.
        - min_sz (int): The minimum size threshold for small images.

        Returns:
        - List[Tuple[int, int, int, int]]: A list of tuples representing positions of small images.
        """
        small_image_positions = []
        height, width = image.shape[:2]
        for y in range(0, height, min_sz):
            for x in range(0, width, min_sz):
                # Calculate the coordinates of the small image
                x1, y1 = x, y
                x2, y2 = min(x + min_sz, width), min(y + min_sz, height)
                # Check if the small image contains enough points (threshold)
                tot = np.sqrt(np.sum(image[y1:y2, x1:x2]) / 765)
                if tot >= SEL_THRESHOLD:
                    # print((x, y, min_sz, min_sz), tot)
                    small_image_positions.append((x, y, min_sz, min_sz))
        self.valid_images = small_image_positions
        return small_image_positions

    def apply_preprocessing_filters(self, image: np.ndarray):  # -> np.ndarray:
        """
        Apply preprocessing filters to the image.

        Parameters:
        - image (np.ndarray): The image to be preprocessed.

        Returns:
        - np.ndarray: The preprocessed image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 5, 75, 75)
        edges = cv2.Canny(bilateral, 30, 120)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        orb = cv2.ORB_create(nfeatures=1500)
        featured_image = closing
        # keypoints, descriptors = orb.detectAndCompute(closing, None)
        # featured_image = cv2.drawKeypoints(closing, keypoints, None)
        cv2.imwrite('OutputImg/CrackDetected-Curr.jpg', featured_image)
        return featured_image

    def smooth_image(self, image):
        bilateral = cv2.bilateralFilter(image, 5, 75, 75)
        return bilateral

    def detect_edges(self, image):
        edges = cv2.Canny(image, 30, 120)
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return closing

    def detect_features(self, image):
        orb = cv2.ORB_create(nfeatures=1500)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        featured_image = cv2.drawKeypoints(image, keypoints, None)
        return featured_image

    ###

    def window_crack_calculation(self, x, y, width, height):
        """
        Perform fixed window crack width calculation on a subimage defined by its coordinates.

        Parameters:
        - x (int): The x-coordinate of the top-left corner of the subimage.
        - y (int): The y-coordinate of the top-left corner of the subimage.
        - width (int): The width of the subimage.
        - height (int): The height of the subimage.

        Returns:
        - float: The average angle of lines detected in the subimage.
        """
        img_array = pygame.surfarray.array3d(self.preprocessed_image)
        img = img_array[x:x + width, y:y + height]
        # color_array = np.array([255, 255, 255])  # Assuming white color for line detection
        pointsX = []
        pointsY = []
        szx, szy, _color = np.shape(img)
        for wy in range(szy):
            for wx in range(szx):
                pixel_value = img[wx, wy]
                if np.sum(pixel_value) > 10:  # np.array_equal(pixel_value, color_array):
                    pointsX.append(wx / (szx - 1))
                    pointsY.append(wy / (szy - 1))

        if pointsX == []:
            return None, None, None
        if max(pointsX) - min(pointsX) < 0.8 * (max(pointsY) - min(pointsY)):
            rot90 = True
            aux = pointsX
            pointsX = pointsY
            pointsY = aux
        else:
            rot90 = False
        # points = np.array(points)
        # x = points[:, 0].reshape(-1, 1)
        x = np.array(pointsX)
        idx = x.argsort()
        x = x[idx]
        x0 = x[0]
        x1 = x[-1]
        # y = points[:, 1].reshape(-1, 1)
        y = np.array(pointsY)[idx]
        #if (len(x) > 10):
            #print("x:", x[:5], x[-5:], "\ny:", y[:5], y[-5:])
        #else:
            #print("x:", x, "\ny:", y)

        x = x.reshape(-1, 1)
        # Fit linear regression model
        lrm = LinearRegression()
        lrm.fit(x, y)

        # Extract slope, intercept, and residuals
        slope = lrm.coef_[0]

        intercept = lrm.intercept_
        residuals = np.mean((lrm.predict(x) - y) ** 2)

        # Calculate distance from each point to the regression line
        distances = np.abs(slope * x - y + intercept) / np.sqrt(slope ** 2 + 1)

        # Compute median line thickness
        line_thickness = np.median(distances)

        return slope, intercept, rot90, x0, x1, line_thickness, residuals

    def draw_regression_line(self, m, b, rot90, x0, x1, thickness, x, y, width, height):
        """
        Draw the regression line on the Pygame screen surface.

        Parameters:
        - screen (pygame.Surface): The Pygame screen surface.
        - m (float): The slope of the regression line.
        - b (float): The y-intercept of the regression line.
        - x0,x1 (float) : line segment start and end
        - x (int): The x-coordinate of the top-left corner of the window.
        - y (int): The y-coordinate of the top-left corner of the window.
        - width (int): The width of the window.
        - height (int): The height of the window.
        - sz (int): The size of the window enlargement.
        """
        # Calculate endpoints of the regression line within the window

        if m is None:
            return
        if rot90:
            aux = x
            x = y
            y = aux
            aux = width
            width = height
            height = aux

        xs = x + int(x0 * width)
        ys = y + int((x0 * m + b) * height)
        xe = x + int(x1 * width)
        ye = y + int((x1 * m + b) * height)

        # Draw the regression line on the screen surface
        # print("L:",m,b,rot90,x0,x1,xs,ys,xe,ye)

        if rot90:
            thickness = int(thickness * height)
            #print("DRAW ROT90" + str(m))
            # pygame.draw.line(self.screen, (150, 205, 50), (ys, xs), (ye, xe), 2)
            self.draw_line_str(f"{m:.2f} ({thickness})", (150, 205, 50), (ys, xs), (ye, xe), 2)  #
        else:
            # pygame.draw.line(self.screen,  (150, 205, 50), (xs, ys), (xe, ye), 2)
            thickness = int(thickness * width)
            #print("DRAW NOT ROT90" + str(m))
            self.draw_line_str(f"{m:.2f} ({thickness})", (150, 205, 50), (xs, ys), (xe, ye), 2)

    def setup_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.image.shape[1], self.image.shape[0]))
        # Load a standard font (Arial) on macOS
        font_path = "/System/Library/Fonts/Geneva.ttf"  ##"/System/Library/Fonts/Arial.ttf"  # Path to Arial font on macOS
        font_size = 10
        self.font = load_font(font_path, font_size)

        pygame.display.set_caption("Original Image")
        self.image_surf = pygame.image.frombuffer(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), self.image.shape[1::-1],
                                                  "RGB")
        self.screen.blit(self.image_surf, (0, 0))
        pygame.display.flip()
        pygame.display.set_caption("Press 'r' to draw a thick yellow rectangle")

    def draw_line_str(self, str, number_color, start, end, lw):
        """
        Draw a line with a number near its midpoint on the Pygame screen surface.
    
        Parameters:
        - screen (pygame.Surface): The Pygame screen surface.
        - start (tuple): The (x, y) coordinates of the starting point of the line.
        - end (tuple): The (x, y) coordinates of the ending point of the line.
        - number (int): The number to display near the midpoint of the line.
        - font (pygame.font.Font): The Pygame font object to use for rendering the number.
        - number_color (tuple, optional): The RGB color of the number. Defaults to red (255, 0, 0).
        """
        # Draw the line
        pygame.draw.line(self.screen, number_color, start, end, lw)

        # Calculate the midpoint of the line
        midpoint = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)

        # Render the number text
        number_text = self.font.render(str, True, number_color)

        # Get the size of the rendered text
        text_width, text_height = number_text.get_size()

        # Calculate the position to display the number near the midpoint (above the line)
        text_pos = (midpoint[0] - text_width // 2, midpoint[1] - text_height // 2 - 10)

        # Display the number near the midpoint
        self.screen.blit(number_text, text_pos)

    def display_image(self):
        """
        Display the original image using Pygame.
        """

        if self.screen is None:
            self.setup_pygame()

        rectangle_draging = False

        # object current co-ordinates 
        # x = 200
        # y = 200

        # dimensions of the object 
        # width = 20
        # height = 20

        vel = 3  ## speed of change with keyboard

        small_pos = 0  ## Current subposition
        # Set up colors

        show = {'valid_positions': "", 'preprocessed_image': False, 'cursor': False}
        ## what should be shown
        show = self.draw_scene(show)
        pygame.display.flip()

        while True:
            pygame.time.delay(50)
            refresh = False

            # handle quit and mouse events (keyboard is handled later)
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    # pygame.display.quit()
                    self.screen = None
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # if rectangle.collidepoint(event.pos):
                    rectangle_draging = True
                    mouse_down = event.pos
                    refresh = True
                    if show['cursor']:
                        ## mouse down only drags on bottomright (else resets start)
                        if (abs(mouse_down[0] - self.rectangle.bottomright[0]) > 10 and \
                                abs(mouse_down[1] - self.rectangle.bottomright[1]) > 10):
                            self.rectangle.x = mouse_down[0]
                            self.rectangle.y = mouse_down[1]
                        # (abs(mouse_down[0] - self.rectangle.topleft[0]) < 10 and \
                        # abs(mouse_down[1] - self.rectangle.topleft[1]) < 10 ) or \
                        else:
                            pass
                            # mouse_dragging = False

                    else:
                        show['cursor'] = True
                        self.rectangle.x = mouse_down[0]
                        self.rectangle.y = mouse_down[1]
                        self.rectangle.width = 4
                        self.rectangle.height = 4



                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        rectangle_draging = False
                        mouse_up = event.pos
                        ## if cursor is not visible it becomes visible
                        ##  or if it is not dragged
                        if not (show['cursor']) or \
                                abs(mouse_up[0] - mouse_down[0]) < 3 and \
                                abs(mouse_up[1] - mouse_down[1]) < 3:
                            self.rectangle.x = mouse_up[0]
                            self.rectangle.y = mouse_up[1]
                            show['cursor'] = True
                            refresh = True

                if event.type == pygame.MOUSEMOTION:
                    if rectangle_draging:
                        mouse_x, mouse_y = event.pos
                        offset_x = abs(self.rectangle.x - mouse_x)
                        offset_y = abs(self.rectangle.y - mouse_y)

                        if offset_x > 3:
                            self.rectangle.width = offset_x
                            refresh = True
                        if offset_y > 3:
                            self.rectangle.height = offset_y
                            refresh = True
                        # show['cursor'] = True

            if (not (refresh)):
                keys = pygame.key.get_pressed()

                refresh = True
                if keys[pygame.K_q]:
                    print("interface active: image_processor.display_image() to restart")
                    return
                # Cursor commands
                elif keys[pygame.K_LEFT] and self.rectangle.x > vel:
                    self.rectangle.x -= vel
                    show['cursor'] = True
                elif keys[pygame.K_RIGHT] and self.rectangle.x < self.iwidth - vel:
                    self.rectangle.x += vel
                    show['cursor'] = True
                elif keys[pygame.K_UP] and self.rectangle.y > 0:
                    self.rectangle.y -= vel
                    show['cursor'] = True
                elif keys[pygame.K_DOWN] and self.rectangle.y < self.iheight - vel:
                    self.rectangle.y += vel
                    show['cursor'] = True
                elif keys[pygame.K_l] and self.rectangle.y < self.iheight - 1 and \
                        self.rectangle.x < self.iwidth - 1:
                    self.rectangle.height += 1
                    self.rectangle.width += 1
                    show['cursor'] = True
                elif keys[pygame.K_k] and \
                        self.rectangle.height > 3 and \
                        self.rectangle.width > 3:
                    self.rectangle.height -= 1
                    self.rectangle.width -= 1
                    show['cursor'] = True
                ## cursor to sub-image
                elif keys[pygame.K_j] and self.valid_positions is not None:
                    show['cursor'] = True
                    self.rectangle = pygame.rect.Rect(self.valid_positions[small_pos])
                    small_pos += 1
                    if (small_pos >= len(self.valid_positions)):
                        small_pos = 0
                    pygame.time.delay(50)
                elif keys[pygame.K_h] and self.valid_positions is not None:
                    show['cursor'] = True
                    self.rectangle = pygame.rect.Rect(self.valid_positions[small_pos])
                    small_pos -= 1
                    if (small_pos < 0):
                        small_pos = len(self.valid_positions) - 1
                    pygame.time.delay(50)
                    ## other
                elif keys[pygame.K_v]:
                    if show['valid_positions'] == "l":
                        show['valid_positions'] = ""
                    elif show['valid_positions'] == "m":
                        show['valid_positions'] = "l"
                    else:
                        show['valid_positions'] = "m"

                    pygame.time.delay(500)
                elif keys[pygame.K_p]:
                    show['preprocessed_image'] = not show['preprocessed_image']
                    pygame.time.delay(500)
                elif keys[pygame.K_r]:
                    if not (show['cursor']):
                        show['cursor'] = True
                        pygame.time.delay(500)
                    else:
                        show['cursor'] = False
                        pygame.time.delay(500)
                else:
                    refresh = False

            if keys[pygame.K_f] and show['cursor']:
                (x, y, width, height) = self.rectangle
                (m, b, rot90, x0, x1, thick, res) = self.window_crack_calculation(x, y, width, height)

                self.draw_regression_line(m, b, rot90, x0, x1, thick, x, y, width, height)
                # print("Regression:", m,b,res)
                pygame.display.flip()
                pygame.time.delay(1000)

                cv2.rectangle(self.image, (x, y), (x + width, y + height), (255, 0, 0), 2)  # Azul com espessura 2

                res_text = float(res)
                res_text = round(res_text, 4)
                # Configurações do texto
                fonte = cv2.FONT_HERSHEY_SIMPLEX
                escala_fonte = 0.3
                cor_texto = (255, 0, 0)  # Branco
                espessura_texto = 1

                # Calcular a posição do texto para centralizar dentro do retângulo
                (tamanho_texto, baseline) = cv2.getTextSize(str(res_text), fonte, escala_fonte, espessura_texto)
                x_texto = x + (width - tamanho_texto[0]) // 2
                y_texto = y + (height + tamanho_texto[1]) // 2

                # Imprimir o texto na imagem
                cv2.putText(self.image, str(res_text), (x_texto, y_texto), fonte, escala_fonte, cor_texto,
                            espessura_texto)

                # Visualizar a imagem com Matplotlib
                #plt.imshow(self.image)
                plt.axis('off')  # Ocultar os eixos
                # plt.show()
                plt.savefig("res_windows.pdf")

                refresh = False
            if refresh:
                show = self.draw_scene(show)
                pygame.display.flip()
                refresh = False

    def draw_scene(self, show: dict) -> None:
        """
        Draw the scene with the original image, valid positions, and/or preprocessed image.
    
        Parameters:
        - screen: The Pygame screen surface.
        - image_surf: The Pygame image surface of the original image.
        - show_valid_positions (bool): Whether to display valid positions.
        - show_preprocessed_image (bool): Whether to display the preprocessed image.
        """
        YELLOW = (255, 255, 0)
        if self.screen is None:
            print("pygame not started")
            return show
        self.screen.fill((0, 0, 0))

        if show['preprocessed_image']:
            self.draw_preprocessed_image(self.screen)
        else:
            self.screen.blit(self.image_surf, (0, 0))

        if show['valid_positions'] != "":
            self.draw_valid_positions(self.screen, show['valid_positions'])

        if show['cursor']:
            # win.fill((0, 0, 0))
            pygame.draw.rect(self.screen, YELLOW, self.rectangle, width=4)
            #print("Pos:", self.rectangle)

            # pygame.draw.rect(self.screen, YELLOW, (100, 100, 200, 100), 3)
            # show['cursor'] = None
        return show

    def draw_valid_positions(self, screen, type):
        """
        Draw rectangles for valid positions on the screen.
    
        Parameters:
        - screen: The Pygame screen surface.
        """
        #print("showing valid positions")
        for position in self.valid_positions:
            if type == "l":
                (x, y, w, h) = position
                (m, b, rot90, x0, x1, thick, res) = self.window_crack_calculation(x, y, w, h)
                degrees = math.degrees(np.arctan(m))
                #if rot90: degrees = degrees + 90
                rot_degrees = degrees
                if m >= 0:
                    rot_degrees = -degrees

                self.declives.append(degrees)
                self.draw_regression_line(m, b, rot90, x0, x1, thick, x, y, w, h)

                res_window_object = R_WINDOW(position, res, rot_degrees)
                r_windows.append(res_window_object)


            else:
                pygame.draw.rect(screen, (0, 255, 0), position, 1)

    def draw_res_windows(self):

        plt.clf()
        plt.figure(figsize=(8, 6), dpi=100)

        for r_w in r_windows:
            (x, y, w, h) = r_w.window
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Azul com espessura 2

            res_text = float(r_w.res)
            res_text = round(res_text, 4)
            # Configurações do texto
            fonte = cv2.FONT_HERSHEY_SIMPLEX
            escala_fonte = 0.3
            cor_texto = (255, 0, 0)  # Branco
            espessura_texto = 1

            # Calcular a posição do texto para centralizar dentro do retângulo
            (tamanho_texto, baseline) = cv2.getTextSize(str(res_text), fonte, escala_fonte, espessura_texto)
            x_texto = x + (w - tamanho_texto[0]) // 2
            y_texto = y + (h + tamanho_texto[1]) // 2

            # Imprimir o texto na imagem
            cv2.putText(self.image, str(res_text), (x_texto, y_texto), fonte, escala_fonte, cor_texto, espessura_texto)

            # Visualizar a imagem com Matplotlib
            #plt.imshow(self.image)
            plt.axis('off')  # Ocultar os eixos
        # plt.show()
        plt.savefig("res_windows.pdf")

    def draw_preprocessed_image(self, screen):
        """
        Draw the preprocessed image on the screen.
    
        Parameters:
        - screen: The Pygame screen surface.
        """
        if self.preprocessed_image is None:
            print("No preproc image to show...")
            return

        #print("showing preproc image")
        # preprocessed_surf = pygame.image.frombuffer(cv2.cvtColor(self.full_preprocessed_image, cv2.COLOR_BGR2RGB), self.full_preprocessed_image.shape[1::-1], "RGB")
        # self.preprocessed_image = cv2.imread('OutputImg/CrackDetected-Curr.jpg')
        screen.blit(self.preprocessed_image, (0, 0))

    def histogram(self):
        declives = image_processor.declives
        print(len(declives))
        print(declives)

        media = sum(declives) / len(declives)
        print('Média Declives:', media)

        positivos = [declive for declive in declives if declive >= 0]
        print('Orientação positiva:', len(positivos))

        negativos = [declive for declive in declives if declive < 0]
        print('Orientação negativa:', len(negativos))

        declives = [np.round(x, 2) for x in declives]
        print(declives)

        counts, bins, patches = plt.hist(declives, bins=20, edgecolor='black')

        plt.xticks(bins[::5], bins[::5], rotation=90)

        plt.title('Histograma de Declives')
        plt.xlabel('Declives')
        resultado = f'Média: {media}\n Orientação positiva: {len(positivos)}\nOrientação negativa: {len(negativos)}'
        plt.text(0.5, 0.95, resultado, transform=plt.gca().transAxes, verticalalignment='top', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig("hist.pdf", bbox_inches='tight')
        plt.savefig("hist.pdf")


# Usage example
if __name__ == "__main__":
    image_name = "9.jpeg"
    comp_real_qr = 5
    largura_real_qr = 5
    qr_area, qr_area_real, qr_width = utils.calc_pixels_e_area_qrcode(image_name, comp_real_qr=comp_real_qr, largura_real_qr=largura_real_qr)
    image_processor = ImageProcessor('/Users/raquelpena/Downloads/projeto_falha-2/resultados_rede/9.jpg')
    image_processor.preprocess_image()
    image_processor.display_image()
    image_processor.histogram()
    image_processor.draw_res_windows()

    window_average_width = pixelcount.calc_pixels_window(image_processor.original_image, r_windows[0].window, r_windows[0].declive, 1)

    # pixel correlation
    real_window_average_width = (window_average_width * largura_real_qr)/qr_width

    #convert to mm
    real_window_average_width = real_window_average_width*10

    print(str(real_window_average_width) + " mm")
#    image_processor = ImageProcessor('resultados_rede/9.jpg')
#    processed_images = image_processor.preprocess_image(min_sz=10)
#    image_processor.test_all()
# Further processing or analysis using image_processor.preprocessed_images,
# image_processor.valid_positions, and image_processor.full_preprocessed_image
