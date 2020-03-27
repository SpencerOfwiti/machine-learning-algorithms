# import libraries
import math, random, time, copy, multiprocessing, itertools, sys, PIL
import networkx as nx
import matplotlib.pyplot as plt
import IPython, io, os
import numpy as np
from urllib.request import urlopen
from io import BytesIO
import pygame
from pygame.locals import *
from PIL import Image

# create the game as an 800 pixel by 800 pixel screen onto which we'll paint the game
window = pygame.display.set_mode((800, 800), pygame.SRCALPHA, depth=32)
images = []


# saving a screenshot of the game to images list
def display_img(data):
    """
    Save screen image to images list
    :param data:
    :return null:
    """
    im = Image.frombytes('RGBA', (800, 800), data)
    bio = BytesIO()
    im.save(bio, format='png')
    images.append(im)


# downloading the blasters image
img = urlopen("https://raw.githubusercontent.com/crash-course-ai/lab3-games/master/assets/blast.png").read()
image_file = BytesIO(img)
blaster_img = pygame.image.load(image_file).convert_alpha()


# defining the blaster object
class PlayerBlaster:
    """
    
    """
    def __init__(self, game, position:pygame.Vector2, move_direction:pygame.Vector2):
        """
        Create a new blast object inside the game, with position and direction
        :param game:
        :param position:
        :param move_direction:
        """
        self.game = game
        self.position = position
        self.rotation = 0.0
        self.radius = 2.0
        self.scale = 0.5
        self.draw_texture = pygame.transform.rotozoom(blaster_img, self.rotation, self.scale)
        self.move_direction = move_direction
        self.move_speed = 500
        self.passed_time = 0.0
        self.life_time = 0.5

        # add this new object to the list of blaster objects in the game
        self.game.add_player_blaster(self)

    def update(self, dt):
        """
        Every time the game updates, the blast needs to move from its current
        position to a new position according to its speed and direction
        :param dt:
        :return null:
        """
        if self.move_direction.length_squared() != 0:
            self.move_direction.scale_to_length(self.move_speed)
        self.position += self.move_direction * dt

        # the wrap_coords function makes the game 'toroidal', which means that the
        # blaster will wrap around from one side of the game to the other
        TrashBlaster.wrap_coords(self.position)
        self.passed_time += dt

        # if the bullet has not hit anything after some time, remove it
        if self.passed_time >= self.life_time:
            self.destroy()

    def render(self, surface:pygame.Surface):
        """
        Draw the blast image at the new location
        :param surface:
        :return null:
        """
        self.draw_texture = pygame.transform.rotozoom(blaster_img, self.rotation, self.scale)
        surface.blit(self.draw_texture, self.position - pygame.Vector2(self.draw_texture.get_rect().size) / 2.0)

    def destroy(self):
        """
        Remove the blast from the game
        :return null:
        """
        self.game.remove_player_blaster(self)

    def get_hit(self):
        """
        Action to perform when blast hits something
        :return null:
        """
        self.destroy()


# Download the John Green Bot image
img = urlopen("https://raw.githubusercontent.com/crash-course-ai/lab3-games/master/assets/john_green_bot.png").read()
image_file = BytesIO(img)
john_green_bot_img = pygame.image.load(image_file).convert_alpha()


# Define the bot object
class JohnGreenBot:
    def __init__(self, game):
        """
        Create a new JohnGreenBot object, add it to the game
        :param game:
        """
        self.game = game
        self.position = pygame.Vector2(400, 400)
        self.rotation = 45.0
        self.radius = 18.0
        self.scale = 0.5
        self.draw_texture = pygame.transform.rotozoom(john_green_bot_img, self.rotation, self.scale)
        self.move_direction = pygame.Vector2(0, 0)
        self.move_speed = 100
        self.is_blasting = False
        self.shoot_accum = 0.0
        self.shoot_period = 0.25

    def update(self, dt):
        """
        Move the object. Check if blasting. This function is called at every game state
        :param dt:
        :return null:
        """
        if self.move_direction.length_squared() != 0:
            self.move_direction.scale_to_length(self.move_speed)
        self.position += self.move_direction * dt
        TrashBlaster.wrap_coords(self.position)
        self.shoot_accum += dt
        self.blast()

    def blast(self):
        """
        Check if blasting. Create a new blaster object if so
        :return blast_state:
        """
        # if not blasting
        if not self.is_blasting or self.shoot_accum < self.shoot_period:
            return False

        # if blasting
        self.shoot_accum = 0
        shoot_direction = pygame.Vector2(-math.cos(math.radians(self.rotation)), math.sin(math.radians(self.rotation)))
        blaster = PlayerBlaster(self.game, self.position + shoot_direction * self.radius, shoot_direction)

        # record blast in the game state
        self.game.blasts += 1
        return True

    def render(self, surface:pygame.Surface):
        """
        Draw John Green Bot at the new location
        :param surface:
        :return null:
        """
        self.draw_texture = pygame.transform.rotozoom(john_green_bot_img, self.rotation, self.scale)
        surface.blit(self.draw_texture, self.position - pygame.Vector2(self.draw_texture.get_rect().size) / 2.0)

    def get_hit(self):
        """
        Perform action when John Green Bot is hit
        :return null:
        """
        self.game.lose()


# Download the trash image
img = urlopen("https://raw.githubusercontent.com/crash-course-ai/lab3-games/master/assets/trash.png").read()
image_file = BytesIO(img)
trash_img = pygame.image.load(image_file).convert_alpha()


# defining the trash object
class Trash:
    def __init__(self, game, position:pygame.Vector2, move_direction:pygame.Vector2, size):
        """
        Create trash object. Add it to the game with position and direction. Give it random speed
        :param game:
        :param position:
        :param move_direction:
        :param size:
        """
        self.game = game
        self.size = size
        self.position = position
        self.rotation = 0.0
        self.radius = 18.0 * 1.3**size
        self.scale = 0.20 * 1.3**size
        self.draw_texture = pygame.transform.rotozoom(trash_img, self.rotation, self.scale)
        self.move_direction = move_direction
        self.move_speed = 150 * 0.9**size * random.uniform(0.75, 1.0)
        self.game.add_trash(self)

    def update(self, dt):
        """
        Move the trash
        :param dt:
        :return null:
        """
        if self.move_direction.length_squared() != 0:
            self.move_direction.scale_to_length(self.move_speed)
        self.position += self.move_direction * dt

        # Make sure that trash behaves toroidally
        TrashBlaster.wrap_coords(self.position)

    def render(self, surface:pygame.Surface):
        """
        Draw the trash
        :param surface:
        :return null:
        """
        self.draw_texture = pygame.transform.rotozoom(trash_img, self.rotation, self.scale)
        surface.blit(self.draw_texture, self.position - pygame.Vector2(self.draw_texture.get_rect().size) / 2.0)

    def destroy(self):
        """
        Remove the trash
        :return null:
        """
        self.game.remove_trash(self)

    def split(self):
        """
        Split the trash into two pieces
        :return null:
        """
        rotate_amount = random.uniform(5, 15)

        # Create two new pieces of trash with slightly smaller size
        Trash(self.game, copy.copy(self.position), self.move_direction.rotate(rotate_amount), self.size - 1)
        Trash(self.game, copy.copy(self.position), self.move_direction.rotate(-rotate_amount), self.size - 1)

        # remove the original trash
        self.destroy()

    def get_hit(self):
        """
        What to do if the trash gets hit by the blaster
        :return null:
        """
        self.game.hits += 1
        if self.size >= 2:
            self.split()
        self.destroy()


# Download the background image
img = urlopen("https://raw.githubusercontent.com/crash-course-ai/lab3-games/master/assets/background_nogradient.png").read()
image_file = BytesIO(img)
background_img = pygame.image.load(image_file).convert_alpha()


# Defining the background
class Background:
    def __init__(self, game):
        """
        Create the background object
        :param game:
        """
        self.game = game
        self.background_img = background_img

    def render(self, surface:pygame.Surface):
        """
        Render the background
        :param surface:
        :return null:
        """
        surface.blit(self.background_img, pygame.Vector2(0, 0))


# defining the scoreboard
class Scoreboard:
    def __init__(self, game):
        """
        Create the scoreboard object
        :param game:
        """
        self.game = game

    def render(self, surface:pygame.Surface):
        """
        Print the score at the top of the game
        :param surface:
        :return null:
        """
        text = str(round(self.game.score, 2)).rjust(6)
        display_font = pygame.font.Font(pygame.font.match_font("Consolas,Lucida Console,Mono,Monospace,Sans"), 20)
        text_image = display_font.render(text, True, (255, 255, 255))
        surface.blit(text_image, pygame.Vector2(0, 0))


# defining the trashblaster game
class TrashBlaster:
    def __init__(self):
        """
        Create new TrashBlaster game with JohnGreenBot, Scoreboard, PlayerABlasters and Trash.
        """
        self.john_green_bot = JohnGreenBot(self)
        self.scoreboard = Scoreboard(self)
        self.background = Background(self)
        self.player_blasters = []
        self.trash_list = []
        self.score = 0
        self.specimen = None
        self.do_render = True
        self.is_playing = True
        self.play_time = 0.0
        self.t_accum = 0.0
        self.spawn_accum = 0.0
        self.blasts = 0
        self.hits = 0

    @staticmethod
    def get_user_rotation(john_green_bot):
        to_mouse = pygame.mouse.get_pos() - john_green_bot.position
        angle = math.atan2(to_mouse.y, to_mouse.x)
        return -math.degrees(angle)

    @staticmethod
    def wrap_coords(point):
        """
        ake the game toroidal. That is, if an object touches one side of the 800 by 800 pixel game board,
        then move it to the opposite end.
        :param point:
        :return:
        """
        dim = (800, 800)
        while point.x < 0:
            point.x += dim[0]
        while point.x > dim[0]:
            point.x -= dim[0]
        while point.y < 0:
            point.y += dim[1]
        while point.y > dim[1]:
            point.y -= dim[1]
        return point

    def run(self, specimen=None, do_render=True):
        """
        Main game loop. Allow the specimen (one of the trained neural networks) play the game.
        Draw and save the game if do_render is true.
        :param specimen:
        :param do_render:
        :return:
        """
        self.specimen = specimen
        self.do_render = do_render

        # Create a game surface if we choose to render
        if do_render:
            render_surface = pygame.display.get_surface()

        # record starting time
        t1 = time.time()

        # make new trash
        self.create_trash()

        # record number of game frames as count
        count = 0

        # main loop
        while self.is_playing:
            # update the time
            t2 = time.time()
            # dt is the delta-time, the change in time between game states.
            # The purpose of this variable is to allow the game to play as fast as the system can allow
            dt = t2 - t1
            t1 = t2

            if self.specimen and not do_render:
                dt = 0.05

            # ask the John Green Bot - specimen for input, but slow it down to once every 4 frames,
            # to make it a bit more realistic.
            if count % 4 == 0:
                self.apply_input()

            # update the game state
            self.update(dt)
            self.check_collisions()

            # make new trash if needed
            self.create_trash()

            # if we are asking this particular game to be rendered, then take a snapshot of the gameboard and save it.
            if self.do_render:
                self.render(render_surface)
                pygame.display.flip()
                data = pygame.image.tostring(window, 'RGBA')
                display_img(data)
            count += 1

        return self.score

    def apply_input(self):
        """
        Ask the specimen (i.e. John Green Bot's neural network) for an action
        :param self:
        :return:
        """
        self.specimen.apply_input(self)

    def update(self, dt):
        """
        Move the objects in the game by calling their update functions.
        :param dt:
        :return:
        """
        self.john_green_bot.update(dt)
        for player_blaster in self.player_blasters:
            player_blaster.update(dt)
        for trash in self.trash_list:
            trash.update(dt)
        self.play_time += dt
        self.score = self.calc_score()

    def calc_score(self):
        """
        This calculates the score for the game.
        :param self:
        :return:
        """
        # CHANGEME - change the values of this function and watch how John Green Bot's
        # learned behaviour changes in response.
        return self.play_time*1 + self.hits*10 + self.blasts*-2

    def check_collisions(self):
        """
        Check if a blaster hit some trash, or if some trash hit John Green Bot.
        :return:
        """
        to_hit = set()
        for trash in self.trash_list:
            for player_blaster in self.player_blasters:
                if trash.position.distance_squared_to(player_blaster.position) <= (trash.radius + player_blaster.radius)**2:
                    # A hit occurs if one object is within its radius' distance to the other object's radius
                    to_hit.add(trash)
                    to_hit.add(player_blaster)
            if trash.position.distance_squared_to(self.john_green_bot.position) <= (trash.radius + self.john_green_bot.radius)**2:
                to_hit.add(self.john_green_bot)

        # call the get_hit function for all objects that are found to have been hit
        for thing in to_hit:
            thing.get_hit()

    def create_trash(self):
        """
        Create trash with different size as needed
        :return:
        """
        # adjust difficulty (number of trash pieces) as time progresses
        if self.play_time <= 20:
            difficulty = 11
        elif self.play_time <= 40:
            difficulty = 14
        elif self.play_time <= 60:
            difficulty = 16
        elif self.play_time <= 80:
            difficulty = 18
        else:
            difficulty = 20

        while len(self.trash_list) < difficulty:
            size = random.randint(1, 4)
            self.spawn_trash(size)

    def spawn_trash(self, size):
        """
        Draw and place the trash objects randomly on the board
        :param size:
        :return:
        """
        position = pygame.Vector2(0, 0)
        if bool(random.randint(0, 1)):
            if bool(random.randint(0, 1)):
                position = pygame.Vector2(random.uniform(0, 800), 0)
            else:
                position = pygame.Vector2(random.uniform(0, 800), 800)
        else:
            if bool(random.randint(0, 1)):
                position = pygame.Vector2(0, random.uniform(0, 800))
            else:
                position = pygame.Vector2(800, random.uniform(0, 800))

        # give Trash random direction
        move_direction = pygame.Vector2(1, 0).rotate(random.uniform(0, 360))
        Trash(self, position, move_direction, size)

    def render(self, surface:pygame.Surface):
        """
        Draw all the objects in the game
        :param surface:
        :return:
        """
        self.background.render(surface)
        self.john_green_bot.render(surface)
        for player_blaster in self.player_blasters:
            player_blaster.render(surface)
        for trash in self.trash_list:
            trash.render(surface)
        self.scoreboard.render(surface)

    def add_player_blaster(self, blaster):
        self.player_blasters.append(blaster)

    def remove_player_blaster(self, blaster):
        if blaster in self.player_blasters:
            self.player_blasters.remove(blaster)

    def add_trash(self, trash):
        self.trash_list.append(trash)

    def remove_trash(self, trash):
        if trash in self.trash_list:
            self.trash_list.remove(trash)

    def lose(self):
        """
        If we lose the game, then set the main loop condition to false.
        :return:
        """
        self.is_playing = False


# offset for calculating items relative to John Green Bot.
OFFSETS = [
    pygame.Vector2(x, y) for x in [-800, 0, 800] for y in [-800, 0, 800]
]


class Specimen:
    def __init__(self):
        """
        Create a specimen (i.e. one of John Green Bot's brains)

        25 inputs: 5 attributes (x, y, x_vel, y_vel, radius) of nearest trash objects.

        5 outputs: 5 moves (x, y, aim_x, aim_y, blast)
        """
        self.NIMPUTS = 25
        self.NOUTPUTS = 5
        self.NHIDDEN = 1
        self.HIDDENSIZE = 15

        self.input_layer = np.zeros((self.NIMPUTS, self.HIDDENSIZE))
        self.inter_layers = np.zeros((self.HIDDENSIZE, self.HIDDENSIZE, self.NHIDDEN))
        self.output_layer = np.zeros((self.HIDDENSIZE, self.NOUTPUTS))

        self.input_bias = np.zeros((self.HIDDENSIZE))
        self.inter_biases = np.zeros((self.HIDDENSIZE, self.NHIDDEN))
        self.output_bias = np.zeros((self.NOUTPUTS))

        self.input_values = np.zeros((self.NIMPUTS))
        self.output_values = np.zeros((self.NOUTPUTS))

    def activation(self, value):
        """
        Activation function, i.e. when to shoot or move
        :param value:
        :return:
        """
        return 0 if value < 0 else value

    def evaluate(self):
        """
        Calculate the final output values by evaluating the parameters of the specimen.
        Pass output values through activation function.
        :return:
        """
        terms = np.dot(self.input_values, self.input_layer) + self.input_bias
        for i in range(self.NHIDDEN):
            terms = np.array([self.activation(np.dot(terms, self.inter_layers[j, :, i]))
                              for j in range(self.HIDDENSIZE)]) + self.inter_biases[:, i]
        self.output_values = np.dot(terms, self.output_layer) + self.output_bias

    def mutate(self):
        """
        Mutate the parameters of the specimen with a probability of 0.05 using a Gaussian function with standard
        deviation of 1. The gaussian function is important because it allows most mutations to be small, but a few to
        be very large
        :return:
        """
        RATE = 1.0
        PROB = 0.05

        for i in range(self.NIMPUTS):
            for j in range(self.HIDDENSIZE):
                if random.random() < PROB:
                    self.input_layer[i, j] += random.gauss(0.0, RATE)

        for i in range(self.HIDDENSIZE):
            for j in range(self.HIDDENSIZE):
                for k in range(self.NHIDDEN):
                    if random.random() < PROB:
                        self.inter_layers[i, j, k] += random.gauss(0.0, RATE)

        for i in range(self.HIDDENSIZE):
            for j in range(self.NOUTPUTS):
                if random.random() < PROB:
                    self.output_layer[i, j] += random.gauss(0.0, RATE)

        for i in range(self.HIDDENSIZE):
            if random.random() < PROB:
                self.input_bias[i] += random.gauss(0.0, RATE)

        for i in range(self.HIDDENSIZE):
            for j in range(self.NHIDDEN):
                if random.random() < PROB:
                    self.inter_biases[i, j] += random.gauss(0.0, RATE)

        for i in range(self.NOUTPUTS):
            if random.random() < PROB:
                self.output_bias[i] += random.gauss(0.0, RATE)

    def calc_fitness(self, do_render=False):
        """
        This function calculates the fitness (i.e. the smartness) of the specimen by playing the game and returning
        the final score.
        :param do_render:
        :return:
        """
        game = TrashBlaster()
        return game.run(specimen=self, do_render=do_render)

    def min_offset(self, point1, point2):
        """
        Helper function for apply input
        :param point1:
        :param point2:
        :return:
        """
        candidates = (point2 - point1 + v for v in OFFSETS)
        return min(candidates, key=lambda v: v.length_squared())

    def apply_input(self, game):
        """
        This function takes the game state, loads   it into the neural network, computes the output, and performs
        the output actions.
        :param game:
        :return:
        """
        john_green_bot = game.john_green_bot

        offsets = {a: self.min_offset(john_green_bot.position, a.position) for a in game.trash_list}

        trash_list = sorted(game.trash_list, key=lambda a: offsets[a].length_squared())
        visible_trash = []
        if len(trash_list) > 5:
            visible_trash = trash_list[0:4]

        # get all the trash and add them as input to the neural network
        for i in range(len(visible_trash)):
            self.input_values[5 * i + 0] = offsets[visible_trash[i]].x
            self.input_values[5 * i + 1] = offsets[visible_trash[i]].y
            self.input_values[5 * i + 2] = visible_trash[i].move_direction.x if abs(visible_trash[i].move_direction.x) > 0.5 else 0
            self.input_values[5 * i + 3] = visible_trash[i].move_direction.y if abs(visible_trash[i].move_direction.y) > 0.5 else 0
            self.input_values[5 * i + 4] = visible_trash[i].radius

        for i in range(len(visible_trash) * 5, 5 * 5):
            self.input_values[i] = 0.0

        # compute the output
        self.evaluate()

        # actually do the recommended actions
        john_green_bot.move_direction.x = self.output_values[0]
        john_green_bot.move_direction.y = self.output_values[1]
        john_green_bot.rotation = -math.degrees(math.atan2(self.output_values[3], self.output_values[2]))
        john_green_bot.is_blasting = self.output_values[4] > 0.5


def get_fitness(specimen):
    """
    Fitness function used in training.
    :param specimen:
    :return:
    """
    return specimen.calc_fitness()


# Initialize a new game space. This starts everything, creates the surface and initializes all of the
# internal PyGame variables.
pygame.init()

# Let's say that our population size is 1000. That means that there are 1000 different specimen that are competing
# to be in the top quarter.
gen_size = 1000

# We're going to be doing lots of things at the same time. So we're going to create 50 threads to compute
# things simultaneously.
n_threads = 50

# Start up a bunch of processing threads to do lots of work at the same time.
pool = multiprocessing.Pool(n_threads)

# Create gen_size number of NEW specimen.
generation = [Specimen() for i in range(gen_size)]

# The map function applies each specimen in the generation to the get_fitness function. Simply put,
# this next line of code plays 1000 games at the same time and collects their scores.
scores = pool.map(get_fitness, generation)

# Create a map of a specimen to its score.
specimen_score_map = {}
for i in range(len(generation)):
    specimen_score_map[generation[i]] = scores[i]

# Sort the specimen by their score and keep only the top quarter.
quarter_size = gen_size//4
generation = sorted(specimen_score_map, key=lambda k: specimen_score_map[k], reverse=True)[0: quarter_size-1]

# Initialize an interaction variable because the next few cells may be run many times.
iteration = 0

# CHANGEME: How many training iterations should we do on each click?
for _ in range(300):
    # List of game states (images) that we will turn into an animation later.
    images = []

    # Increment the iteration counter
    iteration += 1

    # For each reproducer, create 3 copies, mutate them and add them to the generation.
    for i in range(quarter_size):
        child = copy.deepcopy(generation[i])
        child.mutate()
        generation.append(child)
        child2 = copy.deepcopy(generation[i])
        child2.mutate()
        generation.append(child2)
        child3 = copy.deepcopy(generation[i])
        child3.mutate()
        generation.append(child3)

    # At this point we have a new generation. Quarter of these are parents/reproducers and three-quarters are
    # mutant-children. The pool.map function calls the get_fitness function 'simultaneously' for each specimen
    # in the generation.
    scores = pool.map(get_fitness, generation)
    # Create a map of specimen to it's score.
    specimen_score_map = {}
    for i in range(len(generation)):
        specimen_score_map[generation[i]] = scores[i]

    # Find the mean-average of the scores of all specimen
    average_of_all = sum(specimen_score_map.values())/gen_size

    # Sort the specimen by their score and keep only the top quarter.
    generation = sorted(specimen_score_map, key=lambda k:specimen_score_map[k], reverse=True)[0: quarter_size-1]

    # Find the top example. Call the calc_fitness function with do_render=True so that it fills in the images
    # variable with image-captures of the game play.
    example_score = generation[0].calc_fitness(do_render=True)

    # Find the mean-average of the scores of all reproducers (i.e. the top quarter of all specimen)
    average_of_reproducers = sum(sorted(specimen_score_map.values(), reverse=True)[0: quarter_size])/quarter_size

    # Print the statistics.
    print('ITERATION', iteration)
    print('\tAverage score of all specimen: {:5.3f}'.format(average_of_all))
    print('\tAverage score of reproducers:  {:5.3f}'.format(average_of_reproducers))
    print('\tScore of the video-specimen:   {:5.3f}'.format(example_score))

# pause on the last image
last_image = images[-1]
for i in range(100):
    images.append(last_image)

# Save the images as gif
images[0].save('game_animation.gif',
               save_all=True,
               append_images=images[1:],
               optimize=True,
               duration=10,
               loop=0)

# Open and view the previous top specimen as a video
with open('game_animation.gif', 'rb') as f:
    IPython.display.display(IPython.display.Image(data=f.read(), format='png'))

# visualize the neural network
net = generation[0]
G = nx.DiGraph()

# Build nodes
for i in range(net.input_layer.shape[0]):
    G.add_node('i{}'.format(i), pos=(0, i))
for i in range(net.input_layer.shape[1]):
    G.add_node('h{}'.format(i), pos=(1, 1.5*i+1))
for i in range(net.output_layer.shape[0]):
    G.add_node('h2{}'.format(i), pos=(2, 1.5*i+1))
for i in range(net.output_layer.shape[1]):
    G.add_node('o{}'.format(i), pos=(3, 6*i))

# Draw edges
for i in range(net.input_layer.shape[0]):
    for j in range(net.input_layer.shape[1]):
        G.add_edge('i{}'.format(i), 'h{}'.format(j), weight=net.input_layer[i][j])

for i in range(net.inter_layers.shape[0]):
    for j in range(net.inter_layers.shape[1]):
        G.add_edge("h{}".format(i), 'h2{}'.format(j), weight=net.inter_layers[i][j][0])

for i in range(net.output_layer.shape[0]):
    for j in range(net.output_layer.shape[1]):
        G.add_edge('h2{}'.format(i), 'o{}'.format(j), weight=net.output_layer[i][j])

pos = nx.get_node_attributes(G, 'pos')
edges = []
weights = []
for edge, weight in nx.get_edge_attributes(G, 'weight').items():
    if weight != 0:
        edges.append(edge)
        weights.append(weight)
_ = nx.draw_networkx_nodes(G, pos, node_color='b', alpha=0.5)
_ = nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.Reds)
