import json
import keras
import logging
import numpy as np
import os
import pygame
from typing import Type
from env import Catch  # Assuming Catch class is defined in env.py

# Constants for movements
LEFT = -1
STAY = 0
RIGHT = 1

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

def _save_img(input: Type[np.array], screen, grid_size: int, cell_size: int, score: int) -> None:
    # Draw the environment
    screen.fill((0, 0, 0))  # Clear the screen

    # Reshape the input to match grid_size x grid_size
    input_grid = input.reshape((grid_size, grid_size))

    # Loop through the grid and draw based on the values
    for y in range(grid_size):
        for x in range(grid_size):
            cell_value = input_grid[y, x]
            if cell_value == 1:
                pygame.draw.rect(screen, WHITE, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))
            elif cell_value == 2:
                pygame.draw.rect(screen, RED, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))  # Red for special fruit
            elif cell_value == 3:
                pygame.draw.rect(screen, BLUE, pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size))  # Blue for bomb

    # Render score text
    font = pygame.font.Font(None, 36)  # Default system font, size 36
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))  # Position score text on the top-left corner

    pygame.display.flip()

def run_game(model: Type[keras.Model], grid_size: int, cell_size: int) -> None:
    # Define environment
    env = Catch(grid_size)

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
    pygame.display.set_caption("Catch Game")

    clock = pygame.time.Clock()  # Create a clock object to control frame rate

    total_score = 0
    game_count = 0

    # Run the game continuously
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game_count == 0 or env._is_over():
            if game_count > 0:
                score = env.get_score()
                total_score += score
                logging.info(f"Game {game_count} over. Score: {score}")
                logging.info(f"Total score after {game_count} games: {total_score}")

            game_count += 1
            logging.info(f"Starting game {game_count}")
            env.reset()

        game_over = False
        input_t = env.observe()
        _save_img(input_t, screen, grid_size, cell_size, total_score)  # Pass total_score to _save_img
        
        # Continuous movement loop
        while not game_over and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get next action
            q = model.predict(input_t)
            action = np.argmax(q[0])

            # Apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)

            _save_img(input_t, screen, grid_size, cell_size, total_score)  # Pass total_score to _save_img
            pygame.time.wait(50)  # Adjust game speed here (milliseconds)
            clock.tick(60)  # Limit frame rate to 60 FPS

        pygame.time.wait(500)  # Pause between games

    pygame.quit()  # Quit Pygame when the loop ends

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Ensure folders are created
    model_path = "./model"

    # Make sure this grid size matches the value used for training
    grid_size = int(os.environ.get("TRAIN_GRID_SIZE", 20))
    cell_size = 40  # Set the size of each cell to 40x40 pixels

    # Load the trained model
    with open(model_comp_path := f"{model_path}/model.json", "r") as jfile:
        model = keras.models.model_from_json(json.load(jfile))
    model.load_weights(f"{model_path}/model.h5")
    model.compile("sgd", "mse")
    logging.info(f"Model {model_comp_path} loaded and compiled")

    # Play the game
    run_game(model, grid_size, cell_size)
