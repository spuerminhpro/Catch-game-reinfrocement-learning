

import json
import numpy as np
import logging
import os
from typing import Type

import optuna
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import keras

from experience_replay import ExperienceReplay
from env import Catch

# Define global variables
num_actions = 3
grid_size = 20  # Adjust as needed for your game environment

def define_model(
    hidden_size: int,
    num_actions: int,
    learning_rate: float = 0.01,
    hidden_activation: str = "relu",
    loss: str = "mse",
    hidden_layers: int = 2,
) -> Type[keras.Model]:
    model = Sequential()
    model.add(
        Dense(hidden_size, input_shape=(grid_size**2,), activation=hidden_activation)
    )
    # Dynamically add additional hidden layers
    for _ in range(hidden_layers - 1):
        model.add(Dense(hidden_size, activation=hidden_activation))
    model.add(Dense(num_actions))
    model.compile(optimizer=Adam(learning_rate), loss=loss)
    return model

def train_model(
    model: Type[keras.Model],
    epochs: int,
    experience_replay: object,
    epsilon: float,
    batch_size: int,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.05,
) -> Type[keras.Model]:

    logging.info("Initializing model training")
    win_count = 0

    for epoch in range(epochs):
        env.reset()
        current_state = env.observe()

        loss = 0.0
        game_over = False

        while not game_over:
            previous_state = current_state

            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)[0]
            else:
                q = model.predict(previous_state, verbose=False)
                action = np.argmax(q[0])

            current_state, reward, game_over = env.act(action)

            # Increment win count if the reward is positive (caught regular or special fruit)
            if reward > 0:
                win_count += 1

            experience_replay.add_experience(
                [previous_state, int(action), reward, current_state], game_over
            )

            inputs, targets = experience_replay.get_qlearning_batch(
                model, batch_size=batch_size
            )

            loss += model.train_on_batch(inputs, targets)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        logging.info(
            f"Epoch {epoch + 1}/{epochs} | Loss {loss:.4f} | Score {env.get_score()} | Win count {win_count}/{epoch + 1} | Epsilon {epsilon:.4f}"
        )
    return model


def evaluate_model(model, env, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        env.reset()
        current_state = env.observe()
        game_over = False
        episode_reward = 0

        while not game_over:
            q_values = model.predict(current_state, verbose=False)
            action = np.argmax(q_values[0])
            current_state, reward, game_over = env.act(action)
            episode_reward += reward

        total_reward += episode_reward

    average_reward = total_reward / num_episodes
    return average_reward

def objective(trial):
    # Define the hyperparameters to be optimized
    hidden_size = trial.suggest_int('hidden_size', 50, 200)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    epsilon = trial.suggest_uniform('epsilon', 0.1, 1.0)  # Ensure a higher initial epsilon
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.995, 0.999)  # Use a smaller decay rate
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    # Define and compile your model
    model = define_model(hidden_size, num_actions, learning_rate=learning_rate)

    # Train your model
    trained_model = train_model(model, epochs, exp_replay, epsilon, batch_size, epsilon_decay)

    # Evaluate your model
    average_reward = evaluate_model(trained_model, env, num_episodes=10)

    return average_reward

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info(
        f"Number GPUs available: {len(tf.config.experimental.list_physical_devices('GPU'))}"
    )

    # Create all necessary folders
    model_path = "./model"
    os.makedirs(model_path, exist_ok=True)

    # Define environment variable parameters. Smaller parameters values are adopted to reduce training time.
    epochs = int(os.environ.get("TRAIN_EPOCHS", 1500))
    max_memory = int(os.environ.get("TRAIN_MAX_MEMORY", 4000))
    hidden_layers = int(os.environ.get("TRAIN_HIDDEN_LAYERS", 2))
    discount = float(os.environ.get("DISCOUNT", 1.0))
    warm_start_model = os.environ.get("TRAIN_WARM_START_PATH")

    # Define Environment
    env = Catch(grid_size)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory, discount=discount)

    # Optimize hyperparameters with Optuna
    study = optuna.create_study(direction='maximize')  # 'maximize' if you want to maximize the metric
    study.optimize(objective, n_trials=10)  # Number of trials to run (adjust as needed)

    # Retrieve the best parameters found by Optuna
    best_params = study.best_params
    logging.info(f"Best Parameters: {best_params}")

    # Use the best parameters to train the final model
    best_hidden_size = best_params['hidden_size']
    best_learning_rate = best_params['learning_rate']
    best_epsilon = best_params['epsilon']
    best_batch_size = best_params['batch_size']

    # Define and compile your model with the best parameters
    best_model = define_model(best_hidden_size, num_actions, learning_rate=best_learning_rate)

    # Train your model with the best parameters
    best_trained_model = train_model(best_model, epochs, exp_replay, best_epsilon, best_batch_size)

    # Save trained model weights and architecture
    best_trained_model.save_weights(f"{model_path}/model.h5", overwrite=True)
    with open(f"{model_path}/model.json", "w") as outfile:
        json.dump(best_trained_model.to_json(), outfile)
    logging.info(f"Model saved to path {model_path}")
