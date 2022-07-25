import sys
import pygame as pg
import numpy as np
from neural_network import NeuralNetwork

pg.init()

FPS = 60
PIXEL_SIZE = np.array([26, 26], dtype=np.int32)
SIZE = WIDTH, HEIGHT = PIXEL_SIZE * 28
SCREEN = pg.display.set_mode(SIZE)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CLOCK = pg.time.Clock()
FONT = pg.font.Font("Inconsolata-SemiBold.ttf", 20)
NN = NeuralNetwork()
CANVAS = pg.Surface(SIZE)


is_not_empty = np.zeros(shape=(28, 28), dtype=bool)


NN.load_weights_from_file("weights.txt")
# NN.test()
CANVAS.fill(BLACK)
pg.display.set_caption("Recognize a numeral")

while True:
    for event in pg.event.get():
        if (event.type == pg.QUIT):
            sys.exit()
        elif (event.type == pg.KEYDOWN):
            if (event.key == pg.K_r):
                CANVAS.fill(BLACK)
            # elif (event.key == pg.K_e):
            #     inputs = (pg.surfarray.array2d(pg.transform.scale(SCREEN, (28, 28))) / 2**25)
            #     inputs = inputs.T
            #     inputs = inputs.reshape(1, 28*28)
            #     outputs = NN.execute(inputs)
            #     TEXT = 

        elif (event.type == pg.MOUSEBUTTONUP):
            is_not_empty = np.zeros(shape=(28, 28), dtype=bool)

    
    if (pg.mouse.get_pressed()[0]):
        pos = (np.array(pg.mouse.get_pos(), dtype=np.int32) // PIXEL_SIZE)
        if (not is_not_empty[pos[0], pos[1]]):
            is_not_empty[pos[0], pos[1]] = True
            pos *= PIXEL_SIZE
            pix = pg.Surface(PIXEL_SIZE)
            # pix.set_alpha(255)
            pix.fill(WHITE)
            CANVAS.blit(pix, pos)

            for i in -1, 1:
                pix.set_alpha(128)
                CANVAS.blit(pix, pos + PIXEL_SIZE * np.array([i, 0]))
                CANVAS.blit(pix, pos + PIXEL_SIZE * np.array([0, i]))
                pix.set_alpha(64)
                CANVAS.blit(pix, pos + PIXEL_SIZE * np.array([i, i]))
                CANVAS.blit(pix, pos + PIXEL_SIZE * np.array([i, -i]))

        
    inputs = (pg.surfarray.array2d(pg.transform.scale(CANVAS, (28, 28))) / (2**23)).T.reshape(1, 28*28)
    outputs = NN.execute(inputs)[0]


    SCREEN.fill(BLACK)
    SCREEN.blit(CANVAS, (0, 0))
    for i in range(10):
        out = outputs[i]
        text = f"{i} : {outputs[i]:.3f}"
        col = [out * 255] * 3

        if (col[0] < 80): col = [80] * 3

        text_surface = FONT.render(text, True, col)
        SCREEN.blit(text_surface, (0, i * 20))
        
    pg.display.flip()
    CLOCK.tick(FPS)


