import pygame
import neat
import time
import os
import random
# from itertools import cycle
pygame.font.init()


absolute_path = os.path.dirname(__file__)
BACKGROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join(absolute_path, "Flappy_Bird_assets", "Game Objects", "background-day.png")))
GROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join(absolute_path, "Flappy_Bird_assets", "Game Objects", "base.png")))
PIPE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join(absolute_path, "Flappy_Bird_assets", "Game Objects", "pipe-green.png")))
BIRD_IMAGES = [pygame.transform.scale2x(pygame.image.load(os.path.join(absolute_path, "Flappy_Bird_assets", "Game Objects", "yellowbird-upflap.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join(absolute_path, "Flappy_Bird_assets", "Game Objects", "yellowbird-midflap.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join(absolute_path, "Flappy_Bird_assets", "Game Objects", "yellowbird-downflap.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join(absolute_path, "Flappy_Bird_assets", "Game Objects", "yellowbird-midflap.png")))]
FONT = pygame.font.SysFont('consolas', 30)
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 800
FPS, GAME_SPEED = 60, 5         # FPS regulate how often game "ticks", the other variable regulates "per frame" movement on X axis
WHITE = (255,255,255)

class Bird:
    IMAGES = BIRD_IMAGES
    ROTATION = 20
    ROTATION_SPEED = 5
    ANIMATION = 6
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.frame_count = 0
        self.velocity = 0
        self.img_count = 0
        self.img = self.IMAGES[0]
        
    def jump(self):
        self.velocity = -9
        self.frame_count = 0
        
    def move(self):
        self.frame_count += 1 #tick happened, frame moved by                    #todo needs refining
        # equation of uniformly accelerated motion s = v0*t + (a * t**2)/2      #todo bird animation
        self.y = self.y + self.velocity * self.frame_count + self.frame_count ** 2

    def draw(self, surface):
        self.img = self.IMAGES[(self.img_count // self.ANIMATION % 4)]
        # self.img = cycle([self.IMAGES[0], self.IMAGES[1], self.IMAGES[2], self.IMAGES[3]])

        surface.blit(self.img, self.img.get_rect(topleft = (self.x, self.y)).center)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 200          # how tall is space between pipes for the bird to fly thru
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP  = pygame.transform.flip(PIPE_IMAGE, False, True) # flip the pipe down
        self.PIPE_BOTTOM = PIPE_IMAGE
        self.passed = False     # set to True when bird.X is greater then pipes.X, used for adding new pipe to the list of pipes
        self.make_gap()

    def make_gap(self):
        self.height = random.randrange(50, 450)             # how long is top pipe
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.gap                # bottom pipe starts after the top pipe and gap space

    def move(self):
        self.x -= GAME_SPEED

    def draw(self, surface):
        surface.blit(self.PIPE_TOP, (self.x, self.top))
        surface.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collision_check(self, bird):
        bird_mask = pygame.mask.from_surface(bird.img)              # creates a pixel perfect mask based on image
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        bottom_point = bird_mask.overlap(bottom_mask, bottom_offset) # returns true if the two masks overlap even slightly 
        top_point = bird_mask.overlap(top_mask, top_offset)          # same for the bottom pipe

        if top_point or bottom_point: # if either one is true then we have a collision
            return True
        return False                  # and if not then nothing happened
    
class Ground:
    WIDTH = GROUND_IMAGE.get_width()
    IMAGE = GROUND_IMAGE

    def __init__(self, y):
        self.y = y              # never changes, not important
        self.x1 = 0             # start at the left side of screen
        self.x2 = self.WIDTH    # start after x1

    def move(self):
        self.x1 -= GAME_SPEED
        self.x2 -= GAME_SPEED

        if self.x1 + self.WIDTH < 0:            # x1 off the screen
            self.x1 = self.x2 + self.WIDTH      # x1 jump behind x2
        if self.x2 + self.WIDTH < 0:            # x2 off the screen
            self.x2 = self.x1 + self.WIDTH      # x2 jump behind x1

    def draw(self, surface):
        surface.blit(self.IMAGE, (self.x1, self.y))
        surface.blit(self.IMAGE, (self.x2, self.y))    

def draw_window(surface, birds, pipes, ground, score):
    surface.blit(BACKGROUND_IMAGE, (0, 0))           # background don't move 
    
    for pipe in pipes:                               # draw pipes currently in the list
        pipe.draw(surface)

    for bird in birds:                               # draw birds still in the list
        bird.draw(surface)
        
    ground.draw(surface)                             # draw & move the ground
    text = FONT.render(f'Score: {score}', 1, WHITE)
    surface.blit(text, (SCREEN_WIDTH -10 - text.get_width(), 10))
    pygame.display.update()

def main(genomes, config):
    # bird = Bird(230, 350)
    nets = []
    ge = []
    birds = []

    for _, g in genomes: # loop thru genomes (bird AI's), first is genome ID, second genome object
        net = neat.nn.FeedForwardNetwork.create(g, config)  # set up neural network
        nets.append(net)
        birds.append(Bird(230, 350))    # create corresponding birds
        g.fitness = 0                   # with no starting fitness (reward for effectiveness)
        ge.append(g)

    ground = Ground(730)    # ground at Y position
    pipes = [Pipe(600)]     # first pipe in the list wit a distance to the right
    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) # creating pygame window
    clock = pygame.time.Clock()
    score = 0

    running = True
    while running:              # main game loop
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()
        #bird.move()
        pipe_ind = 0        # pipe indicator indicates which pipe should AI look for checking when to jump
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width(): # once bird.X is greater then pipe.X we switch to the next one
                pipe_ind = 1 # there are at most 2 pipes simultaneously, now we point to he second one index[1]
        else:                # will switch back we we remove pipe from the list
            running = False  # no birds = end loop = end generation
            break

        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1 # genome get fitness per frame (accumulate fitness as long it exists)

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom))) 
            if output[0] > 0.2:     # output basically is genomes decision when to jump
                bird.jump()
        

        add_pipe = False
        rem = []
        for pipe in pipes:                     # loop thru pipes
            for x, bird in enumerate(birds):   # then loop thru birds
                if pipe.collision_check(bird):         # check collision for each one and if true
                    birds.pop(x)               # kill the bird/neural net/genome
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and pipe.x < bird.x: # once bird.X is greater then pipe.X then we add pipe 
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:  # once pipe is completely off screen (left side) then we add new pipe to the list of pipes
                rem.append(pipe)

            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5  # reward in fitness points given to genomes that just passed a pipe without dieing
            pipes.append(Pipe(600)) # we add pipe at xyz distance (to the right)

        for r in rem:
            pipes.remove(r)

        for x, bird in enumerate(birds):        
            if bird.y + bird.img.get_height() >= 730 or bird.y <0: # check if the bird hit the ground or sky
                birds.pop(x)                                       # then kill the stupid bird with its genome and neural net
                nets.pop(x)
                ge.pop(x)
                

            
        ground.move()
        draw_window(window, birds, pipes, ground, score)



#main()

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    winner = population.run(main, 50) # returns last generation (thus 'winner' of the game), can be saved to a file and reused


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt') # configuration file for the NEAT module, important, can mess inside a little bit 
    run(config_path)