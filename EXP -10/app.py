from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2 
import pygame
import sys

# Constants
WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)
BOUNDRYINC = 5
WINDOWSIZEX = 640
WINDOWSIZEY = 480
PREDICT=True
IMAGESAVE=False

# Load Model
MODEL = load_model("foml.h5")

LABELS ={
   0:"ZERO",
   1:"ONE",
   2:"TWO",
   3:"THREE",
   4:"FOUR",
   5:"FIVE",
   6:"SIX",
   7:"SEVEN",
   8:"EIGHT",
   9:"NINE"
}

# Initialize Pygame
pygame.init()
DISPLAYSURF = pygame.display.set_mode((640,480))
pygame.display.set_caption("DIGIT RECOGNITION")
iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1

# Main loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)  
            number_ycord = sorted(number_ycord)  
            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDRYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0), min(number_ycord[-1] + BOUNDRYINC, WINDOWSIZEY)
            number_xcord = []
            number_ycord = []
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            
            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt += 1

            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28)) / 255
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                # Render text
                font = pygame.font.Font(None, 36)
                text_surface = font.render(label, True, RED)
                text_rect_obj = text_surface.get_rect()
                text_rect_obj.left, text_rect_obj.bottom = rect_min_x, rect_max_y
                DISPLAYSURF.blit(text_surface, text_rect_obj)
    
    # Event handling for keys
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)   

    pygame.display.update()
