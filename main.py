import sys
import pygame as pg
import numpy as np
from neural_network import NeuralNetwork

pg.init()

FPS = 512
# PIXEL_SIZE = np.array([26, 26], dtype=np.int32)
RADIUS = 32
SIZE = WIDTH, HEIGHT = 900, 720
SCREEN = pg.display.set_mode(SIZE)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CLOCK = pg.time.Clock()
FONT = pg.font.Font("Inconsolata-SemiBold.ttf", 20)
NN = NeuralNetwork()
CANVAS = pg.Surface(SIZE)


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

    if (pg.mouse.get_pressed()[0]):
        pg.draw.circle(CANVAS, WHITE, pg.mouse.get_pos(), RADIUS, 0)
    elif (pg.mouse.get_pressed()[2]):
        pg.draw.circle(CANVAS, BLACK, pg.mouse.get_pos(), RADIUS, 0)
        
    inputs = (pg.surfarray.array2d(pg.transform.scale(CANVAS, (28, 28))) / (2**20)).T.reshape(1, 28*28)
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


