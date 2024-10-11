class MHGrid:
    def __init__(self, n_rows=4, n_cols=4):
        """
        Initialize the grid environment with the given number of rows and columns.
        """
        # Store grid dimensions
        self.n_rows = n_rows
        self.n_cols = n_cols

        # Map action indices to directions
        self.action_map = {
            0: 'UP',
            1: 'RIGHT',
            2: 'DOWN',
            3: 'LEFT'
        }

        # Initialize the starting position
        self.start_state = (0, 0)

        # Initialize the current state at the starting position
        self.current_state = self.start_state

        # Initialize the history with the starting state
        self.history = [self.current_state]

        # Set the goal position at the bottom-right corner of the grid
        self.goal = (n_rows - 1, n_cols - 1)

    def step(self, action):
        """
        Take a step in the environment based on the action provided.
        If the action would take the agent outside the grid,
        reset the agent to the starting position.
        Returns the next state, reward, and whether the episode is done.
        """
        # Unpack the current state
        row, col = self.current_state

        # Determine the intended new position based on the action
        if action == 0:  # UP
            new_row = row - 1
            new_col = col
        elif action == 1:  # RIGHT
            new_row = row
            new_col = col + 1
        elif action == 2:  # DOWN
            new_row = row + 1
            new_col = col
        elif action == 3:  # LEFT
            new_row = row
            new_col = col - 1
        else:
            raise ValueError("Invalid action.")

        # Check if the new position is outside the grid boundaries
        if (0 <= new_row < self.n_rows) and (0 <= new_col < self.n_cols):
            # Valid move, update the current state
            self.current_state = (new_row, new_col)
        else:
            # Invalid move, reset to starting position
            self.current_state = self.start_state

        # Add the new state to the history
        self.history.append(self.current_state)

        # Check if the goal has been reached
        if self.current_state == self.goal:
            reward = 1  # Positive reward for reaching the goal
            done = True
        else:
            reward = 0  # No reward for other steps
            done = False

        # Return the next state, reward, and done flag
        return self.current_state, reward, done

    def reset(self):
        """
        Reset the environment to the starting state and clear the history.
        """
        # Reset the current state to the starting position
        self.current_state = self.start_state

        # Clear the history and add the starting state
        self.history = [self.current_state]

    def render(self, mode='human'):
        """
        Render the environment.
        Mode can be 'human' or 'ansi':
        - 'human': Display a visualization using matplotlib.
        - 'ansi': Print the environment in the terminal with the current state (*) and the goal (G).
        """
        if mode == 'human':
            # Import matplotlib for visualization
            import matplotlib.pyplot as plt
            import numpy as np

            # Create a grid representation
            grid = np.zeros((self.n_rows, self.n_cols))

            # Mark the current position
            row, col = self.current_state
            grid[row, col] = 0.5  # Agent position

            # Mark the goal position
            goal_row, goal_col = self.goal
            grid[goal_row, goal_col] = 1.0  # Goal position

            # Plot the grid with (0,0) at the upper-left corner
            plt.imshow(grid, cmap='gray_r', interpolation='nearest')
            plt.xticks([])
            plt.yticks([])
            plt.grid(True)
            plt.show()

        elif mode == 'ansi':
            # Create a grid representation using a list of lists
            grid = [[' ' for _ in range(self.n_cols)] for _ in range(self.n_rows)]

            # Mark the goal position
            goal_row, goal_col = self.goal
            grid[goal_row][goal_col] = 'G'

            # Mark the current position
            row, col = self.current_state
            if grid[row][col] == 'G':
                grid[row][col] = '*G'  # Agent is at the goal
            else:
                grid[row][col] = '*'

            # Print the grid from top to bottom to align with (0,0) at the upper-left
            for r in grid:
                print('|' + '|'.join(r) + '|')

        else:
            raise ValueError("Invalid mode for render.")

    def get_state(self):
        """
        Return the current state of the agent in the environment.
        """
        return self.current_state

    def get_history(self):
        """
        Return the history of states visited by the agent.
        """
        return self.history

    def __repr__(self):
        """
        Return a string representation of the environment.
        """
        return f'MHGrid(n_rows={self.n_rows}, n_cols={self.n_cols}, current_state={self.current_state}, goal={self.goal})'

    def animate(self, mode='human', interval=500):
        """
        Replay the history of states visited.
        Mode can be 'human' or 'ansi':
        - 'human': Show an animation using matplotlib.
        - 'ansi': Print the animation in the terminal, updating in place.
        The interval parameter controls the speed of the animation in milliseconds.
        """
        if mode == 'human':
            # Import necessary libraries for animation
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import numpy as np

            # Set up the figure and axis
            fig, ax = plt.subplots()
            grid = np.zeros((self.n_rows, self.n_cols))
            goal_row, goal_col = self.goal
            grid[goal_row, goal_col] = 1.0  # Goal position

            # Display the initial grid
            im = ax.imshow(grid, cmap='gray_r', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True)

            # Function to update the grid for each frame
            def update(frame):
                # Clear the grid
                grid = np.zeros((self.n_rows, self.n_cols))
                # Mark the agent's position
                row, col = self.history[frame]
                grid[row, col] = 0.5  # Agent position
                # Mark the goal position
                grid[goal_row, goal_col] = 1.0
                # Update the image data
                im.set_array(grid)
                return [im]

            # Create the animation
            ani = animation.FuncAnimation(fig, update, frames=len(self.history),
                                          interval=interval, blit=True, repeat=False)
            # Display the animation
            plt.show()

        elif mode == 'ansi':
            # Import time for sleep and os for clearing the terminal
            import time
            import os

            # Iterate through the history of states
            for state in self.history:
                # Clear the terminal
                os.system('cls' if os.name == 'nt' else 'clear')

                # Create a grid representation
                grid = [[' ' for _ in range(self.n_cols)] for _ in range(self.n_rows)]

                # Mark the goal position
                goal_row, goal_col = self.goal
                grid[goal_row][goal_col] = 'G'

                # Mark the current position
                row, col = state
                if grid[row][col] == 'G':
                    grid[row][col] = '*G'  # Agent is at the goal
                else:
                    grid[row][col] = '*'

                # Print the grid from top to bottom
                for r in grid:
                    print('|' + '|'.join(r) + '|')

                # Pause for the specified interval
                time.sleep(interval / 1000.0)
        else:
            raise ValueError("Invalid mode for animate.")

if __name__ == '__main__':
    env = MHGrid()
    print(env)
    print('Current state:', env.get_state())
    env.render(mode='ansi')
    # Example steps
    actions = [1, 1, 1, 1, 1, 1, 1, 2, 3]  # UP, RIGHT, RIGHT, DOWN, LEFT
    for action in actions:
        state, reward, done = env.step(action)
        print(f"Action taken: {env.action_map[action]}")
        print(env)
        print('Current state:', env.get_state())
        env.render(mode='ansi')
        if done:
            print("Goal reached!")
            break
    print('History:', env.get_history())
    env.animate(mode='human', interval=500)
