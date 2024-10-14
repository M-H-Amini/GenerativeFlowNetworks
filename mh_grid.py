class MHGrid:
    def __init__(self, n_rows=4, n_cols=4, goals=None):
        """
        Initialize the grid environment with the given number of rows and columns.

        Args:
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.
            goals (list of tuple): List of goal positions in the grid.
        """
        # Store grid dimensions
        self.n_rows = n_rows
        self.n_cols = n_cols

        # Define terminal state
        self.terminal_state = (n_rows - 1, n_cols)

        # Map action indices to directions (only RIGHT and DOWN)
        self.action_map = {
            0: 'RIGHT',
            1: 'DOWN'
        }

        # Initialize the starting position
        self.start_state = (0, 0)

        # Initialize the current state at the starting position
        self.current_state = self.start_state

        # Initialize the history with the starting state
        self.history = [self.current_state]

        # Set the goal positions
        self.goals = goals if goals is not None else [(n_rows - 1, n_cols - 1)]
        
    def step(self, action):
        """
        Take a step in the environment based on the action provided.
        If the action would take the agent outside the grid,
        move the agent to the terminal state, set done to True, and reward to 1.

        Args:
            action (int): Action to be taken (0: RIGHT, 1: DOWN).

        Returns:
            tuple: (next_state, reward, done)
                next_state (tuple): The next state after taking the action.
                reward (int): Reward received after taking the action.
                done (bool): Whether the episode has ended.
        """
        # Unpack the current state
        row, col = self.current_state

        # Determine the intended new position based on the action
        if action == 0:  # RIGHT
            new_row = row
            new_col = col + 1
        elif action == 1:  # DOWN
            new_row = row + 1
            new_col = col
        else:
            raise ValueError("Invalid action.")

        # Check if the new position is outside the grid boundaries
        if (0 <= new_row < self.n_rows) and (0 <= new_col < self.n_cols):
            # Valid move, update the current state
            self.current_state = (new_row, new_col)
            done = False
            reward = 0

            # Check if the goal has been reached
            if self.current_state in self.goals:
                self.current_state = self.terminal_state
                reward = 1  # Positive reward for reaching the goal
                done = True
        else:
            # Invalid move, move to terminal state
            self.current_state = self.terminal_state
            done = True
            reward = 1  # Reward for reaching terminal state

        # Add the new state to the history
        self.history.append(self.current_state)

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

        Args:
            mode (str): 'human' to display a visualization using matplotlib,
                        'ansi' to print the environment in the terminal with the current state (*) and the goal (G).
        """
        if mode == 'human':
            # Import matplotlib for visualization
            import matplotlib.pyplot as plt
            import numpy as np

            # Create a grid representation
            grid = np.zeros((self.n_rows, self.n_cols))

            # Mark the current position
            if self.current_state != self.terminal_state:
                row, col = self.current_state
                grid[row, col] = 0.5  # Agent position

            # Mark the goal positions
            for goal in self.goals:
                goal_row, goal_col = goal
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

            # Mark the goal positions
            for goal in self.goals:
                goal_row, goal_col = goal
                grid[goal_row][goal_col] = 'G'

            # Mark the current position
            if self.current_state != self.terminal_state:
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

        Returns:
            tuple: The current state.
        """
        return self.current_state

    def get_history(self):
        """
        Return the history of states visited by the agent.

        Returns:
            list of tuple: The list of states visited.
        """
        return self.history

    def legal_actions(self, s=None):
        """
        Get the list of legal actions from a given state.

        Args:
            s (tuple, optional): The state from which to get legal actions. Defaults to current state.

        Returns:
            list of int: List of legal actions (0: RIGHT, 1: DOWN).
        """
        if s is None:
            s = self.current_state
        row, col = s
        actions = []
        if col < self.n_cols - 1:
            actions.append(0)  # RIGHT
        if row < self.n_rows - 1:
            actions.append(1)  # DOWN
        actions = sorted(actions)
        return actions

    def nextState(self, s, a):
        """
        Get the next state from state s taking action a.

        Args:
            s (tuple): The current state.
            a (int): The action taken (0: RIGHT, 1: DOWN).

        Returns:
            tuple: The next state.
        """
        row, col = s
        if a == 0:  # RIGHT
            new_row = row
            new_col = col + 1
        elif a == 1:  # DOWN
            new_row = row + 1
            new_col = col
        else:
            raise ValueError("Invalid action.")
        if (0 <= new_row < self.n_rows) and (0 <= new_col < self.n_cols):
            return (new_row, new_col)
        else:
            # Move to terminal state
            return self.terminal_state

    def parents(self, s):
        """
        Get the list of parent states for a given state.

        Args:
            s (tuple): The state for which to find parent states.

        Returns:
            list of tuple: List of parent states.
        """
        row, col = s
        parents = []
        if col > 0:
            parents.append((row, col - 1))  # LEFT
        if row > 0:
            parents.append((row - 1, col))  # UP
        return parents

    def children(self, s):
        """
        Get the list of child states for a given state.

        Args:
            s (tuple): The state for which to find child states.

        Returns:
            list of tuple: List of child states.
        """
        row, col = s
        children = []
        if col < self.n_cols - 1:
            children.append((row, col + 1))  # RIGHT
        if row < self.n_rows - 1:
            children.append((row + 1, col))  # DOWN
        return children

    def __repr__(self):
        """
        Return a string representation of the environment.

        Returns:
            str: String representation.
        """
        return f'MHGrid(n_rows={self.n_rows}, n_cols={self.n_cols}, current_state={self.current_state}, goals={self.goals})'

    def animate(self, mode='human', interval=500):
        """
        Replay the history of states visited.

        Args:
            mode (str): 'human' to show an animation using matplotlib,
                        'ansi' to print the animation in the terminal, updating in place.
            interval (int): The interval between frames in milliseconds.
        """
        if mode == 'human':
            # Import necessary libraries for animation
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import numpy as np

            # Set up the figure and axis
            fig, ax = plt.subplots()
            grid = np.zeros((self.n_rows, self.n_cols))
            for goal in self.goals:
                goal_row, goal_col = goal
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
                state = self.history[frame]
                if state != self.terminal_state:
                    row, col = state
                    grid[row, col] = 0.5  # Agent position
                # Mark the goal positions
                for goal in self.goals:
                    goal_row, goal_col = goal
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

                # Mark the goal positions
                for goal in self.goals:
                    goal_row, goal_col = goal
                    grid[goal_row][goal_col] = 'G'

                # Mark the current position
                if state != self.terminal_state:
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
    # Example usage
    goals = [(3, 3), (1, 3), (2, 1)]
    env = MHGrid(n_rows=4, n_cols=4, goals=goals)
    print(env)
    print('Current state:', env.get_state())
    env.render(mode='ansi')
    # Example steps
    actions = [0, 0, 0, 1, 1, 0]  # RIGHT, RIGHT, RIGHT, DOWN, DOWN, RIGHT
    actions = [0, 1, 1, 1, 1, 0]  # RIGHT, DOWN, DOWN, DOWN, DOWN, RIGHT
    for action in actions:
        state, reward, done = env.step(action)
        print(f"Action taken: {env.action_map[action]}")
        print(env)
        print('Current state:', env.get_state())
        env.render(mode='ansi')
        if done:
            print("Game over!")
            break
    print('History:', env.get_history())
    env.animate(mode='human', interval=500)
