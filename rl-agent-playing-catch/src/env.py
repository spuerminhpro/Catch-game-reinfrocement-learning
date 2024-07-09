import numpy as np
from typing import Tuple, Type

# Movements
LEFT = -1
STAY = 0
RIGHT = 1

class Catch(object):
    def __init__(self, grid_size: int = 20) -> None:
        self.grid_size = grid_size
        self.reset()
        self.score = 0

    def _update_state(self, action: int) -> None:
        state = self.state
        if action == 0:
            move = LEFT
        elif action == 1:
            move = STAY
        else:
            move = RIGHT

        fruit_row, fruit_col, basket_position = state[0]
        new_basket_position = min(max(1, basket_position + move), self.grid_size - 2)
        fruit_row += 1
        out = np.array([[fruit_row, fruit_col, new_basket_position]])

        assert len(out.shape) == 2
        self.state = out

    def _get_reward(self) -> int:
        fruit_row, fruit_col, basket = self.state[0]

        if fruit_row == self.grid_size - 1:
            if abs(fruit_col - basket) <= 1:
                if self.is_special_fruit(fruit_row, fruit_col):
                    return 10
                elif self.is_bomb(fruit_row, fruit_col):
                    return -5
                else:
                    return 1
            else:
                return -1
        else:
            return 0

    def is_special_fruit(self, fruit_row: int, fruit_col: int) -> bool:
        return self.state[0, 1] == 2

    def is_bomb(self, fruit_row: int, fruit_col: int) -> bool:
        return self.state[0, 1] == 3

    def _is_over(self) -> bool:
        return self.state[0, 0] == self.grid_size - 1

    def observe(self) -> Type[np.array]:
        state = self.state[0]
        canvas = np.zeros((self.grid_size, self.grid_size))
        canvas[state[0], state[1]] = 1
        canvas[-1, state[2] - 1: state[2] + 2] = 1

        if state[1] == 2:
            canvas[state[0], state[1]] = 2
        elif state[1] == 3:
            canvas[state[0], state[1]] = 3

        return canvas.reshape((1, -1))

    def act(self, action: int) -> Tuple[np.array, int, bool]:
        self._update_state(action)
        reward = self._get_reward()
        self.score += reward
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self) -> None:
        m = np.random.randint(1, self.grid_size - 1)

        fruits = [[0, np.random.randint(self.grid_size)]]
        if np.random.uniform() < 0.2:
            unique_fruit_type = np.random.choice([2, 3])
            fruits.append([0, unique_fruit_type])

        if np.random.uniform() < 0.2:
            unique_fruit_type = np.random.choice([2, 3])
            fruits.append([0, unique_fruit_type])

        basket_position = [m]
        self.state = np.array([[fruit[0], fruit[1], basket_position[0]] for fruit in fruits])
        self.score = 0

    def get_score(self) -> int:
        return self.score
