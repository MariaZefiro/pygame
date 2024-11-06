import pygame
import random
import neat
import math

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BRIGHT_RED = (255, 50, 50)  
GREEN = (0, 255, 0)

class Agent:
    def __init__(self):
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.size = 20
        self.speed = 3
        self.direction = random.uniform(0, 2 * math.pi)

    def move(self):
        dx = self.speed * math.cos(self.direction)
        dy = self.speed * math.sin(self.direction)
        new_x = self.x + dx
        new_y = self.y + dy

        if new_x < 0 or new_x + self.size > SCREEN_WIDTH:
            self.direction = math.pi - self.direction
        if new_y < 0 or new_y + self.size > SCREEN_HEIGHT:
            self.direction = -self.direction

        self.x += self.speed * math.cos(self.direction)
        self.y += self.speed * math.sin(self.direction)

    def draw(self):
        pygame.draw.rect(screen, GREEN, (self.x, self.y, self.size, self.size))

    def check_collision(self, obstacles):
        for obstacle in obstacles:
            if (self.x < obstacle.x + obstacle.width and
                self.x + self.size > obstacle.x and
                self.y < obstacle.y + obstacle.height and
                self.y + self.size > obstacle.y):
                return True
        return False

    def get_closest_obstacle(self, obstacles):
        closest_obstacle = None
        min_distance = float('inf')

        for obstacle in obstacles:
            dist_x = obstacle.x - self.x
            dist_y = obstacle.y - self.y
            distance = math.sqrt(dist_x ** 2 + dist_y ** 2)

            if distance < min_distance:
                min_distance = distance
                closest_obstacle = obstacle

        return closest_obstacle

    def avoid_obstacles(self, obstacles):
        closest_obstacle = self.get_closest_obstacle(obstacles)
        if closest_obstacle:
            dist_x = closest_obstacle.x - self.x
            dist_y = closest_obstacle.y - self.y
            distance = math.sqrt(dist_x ** 2 + dist_y ** 2)

            if distance < 100:  
                angle_to_obstacle = math.atan2(dist_y, dist_x)
                avoidance_angle = angle_to_obstacle + math.pi  
                self.direction = avoidance_angle + random.uniform(-0.1, 0.1)  

class Obstacle:
    def __init__(self):
        self.width = 40
        self.height = 40
        self.x = random.randint(0, SCREEN_WIDTH - self.width)
        self.y = random.randint(0, SCREEN_HEIGHT - self.height)
        self.vx = random.choice([-1, 1]) * random.uniform(3, 6)  
        self.vy = random.choice([-1, 1]) * random.uniform(3, 6)

    def move(self):
        self.x += self.vx
        self.y += self.vy
        if self.x <= 0 or self.x + self.width >= SCREEN_WIDTH:
            self.vx *= -1
        if self.y <= 0 or self.y + self.height >= SCREEN_HEIGHT:
            self.vy *= -1

    def draw(self):
        pygame.draw.rect(screen, BRIGHT_RED, (self.x, self.y, self.width, self.height)) 

def eval_genomes(genomes, config):
    clock = pygame.time.Clock()
    generation_number = 1
    for genome_id, genome in genomes:
        genome.fitness = 0
        agent = Agent()
        obstacles = [Obstacle() for _ in range(10)]  
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            inputs = []
            closest_obstacle = agent.get_closest_obstacle(obstacles)
            if closest_obstacle:
                dist_x = closest_obstacle.x - agent.x
                dist_y = closest_obstacle.y - agent.y
                distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
                inputs.append(distance / 100) 

                output = net.activate(inputs)
                agent.direction += (output[0] - 0.5) * 2  

            screen.fill(WHITE)

            agent.avoid_obstacles(obstacles)
            agent.move()

            for obstacle in obstacles:
                obstacle.move()
                obstacle.draw()

            if agent.check_collision(obstacles):
                genome.fitness -= 1  
                running = False
                
            font = pygame.font.Font(None, 36)
            text = font.render(f"Geração: {generation_number}", True, BLACK)
            screen.blit(text, (10, 10))

            agent.draw()
            genome.fitness += 0.1 
            pygame.display.flip()

            clock.tick(30)
            
        generation_number += 1 

def run_neat():
    config_path = "config_neat.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())

    p.run(eval_genomes, 5)

if __name__ == "__main__":
    run_neat()
