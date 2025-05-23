import pygame  # type: ignore
import os
import random
import time
import sys
import numpy as np

pygame.init()
WIN_WIDTH = 500
WIN_HEIGHT = 800
SPEED = 30

BIRD_IMGS = [
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png"))),
]

PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2
        if d >= 16:
            d = 16

        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.height + 50 >= self.y:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count = (self.img_count + 1) % len(self.IMGS)

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME * 2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME * 3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME * 4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME * 4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME * 2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP = 200
    VEL = 5
    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height= random.randrange(50,450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height+self.GAP

    def move(self):
        self.x -=self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset =  (self.x- bird.x, self.top-round(bird.y))
        bottom_offset = (self.x-bird.x, self.bottom-round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if t_point or b_point:
            return True

        return False

class Base:
    VEL = 5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH <0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH <0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

def draw_window(win, bird, pipes , base):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)

    bird.draw(win)
    pygame.display.update()

class GamePlay:
    def __init__(self):
        self.score = 0
        self.bird = Bird(230, 350)
        self.run = True
        self.base = Base(730)
        self.pipes = [Pipe(700)]
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('comicsans', 30)  # Initialize font
        self.frame_iteration = 0
        self.win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))


    def reset(self):
        self.score = 0
        self.bird = Bird(230, 350)  
        self.pipes = [Pipe(700)]
        self.frame_iteration = 0

    def get_state(self):
        pipe_ind = 0
        if len(self.pipes) > 1 and self.bird.x > self.pipes[0].x + self.pipes[0].PIPE_TOP.get_width(): 
            pipe_ind = 1 

        topdistance = abs(self.bird.y - self.pipes[pipe_ind].height)
        bottamdistance = abs(self.bird.y - self.pipes[pipe_ind].bottom)

        state=[
            self.bird.y,
            topdistance,
            bottamdistance
        ]
        return np.array(state, dtype=int)
    
    def play_step(self, action):
        self.frame_iteration +=1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        if np.array_equal(action,[1,0]):
            self.bird.jump()
        reward=0
          
        game_over = False
        add_pipe = False
        rem =[]
        for pipe in self.pipes:
            if pipe.collide(self.bird) :
                game_over=True
                reward=-10
                return reward, game_over, self.score
            
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < self.bird.x:
                pipe.passed = True
                add_pipe = True
            pipe.move()

        if add_pipe:
            self.score += 1
            reward = 10 
            print(self.score) # Update score
            self.pipes.append(Pipe(700))

        for r in rem:
            self.pipes.remove(r)

        if self.bird.y + self.bird.img.get_height() >= 730 :
            game_over=True
            reward=-10
            return reward, game_over, self.score
        
        if self.bird.y < 0:
            game_over=True
            reward=-10
            return reward, game_over, self.score 
        
        self.clock.tick(SPEED)
        self.bird.move()

        self.base.move()
        draw_window(self.win, self.bird, self.pipes, self.base)

        # Render score
        score_text = self.font.render('Score: ' + str(self.score), True, (255, 255, 255))
        # Blit score onto the screen
        self.win.blit(score_text, (10, 10))

        return reward, game_over, self.score


    def dectection(self):

        add_pipe = False
        rem =[]
        for pipe in self.pipes:
            if pipe.collide(self.bird):
                self.reset()
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < self.bird.x:
                pipe.passed = True
                add_pipe = True
            pipe.move()

        if add_pipe:
            self.score += 1 
            print(self.score) # Update score
            self.pipes.append(Pipe(700))

        for r in rem:
            self.pipe_ind=0
            self.pipes.remove(r)

        if self.bird.y + self.bird.img.get_height() >= 730:
            self.run = False

def main():
    
    game = GamePlay()
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    
    while game.run:
        game.clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    game.bird.jump()
        game.dectection()
        game.bird.move()

        game.base.move()
        draw_window(win, game.bird, game.pipes, game.base)

        # Render score
        score_text = game.font.render('Score: ' + str(game.score), True, (255, 255, 255))
        # Blit score onto the screen
        win.blit(score_text, (10, 10))

    pygame.quit()
    quit()

if __name__ == "__main__":
    main()
