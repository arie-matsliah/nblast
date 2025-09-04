import random

PLAYER_X = 'X'
PLAYER_O = 'O'
EMPTY = ' '
WINNING_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),    # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),    # Columns
    (0, 4, 8), (2, 4, 6)                # Diagonals
]


class TicTacToe:  # Environment
    def __init__(self):
        self.board = [EMPTY] * 9
        self.current_player = PLAYER_X

    def reset(self):
        self.board = [EMPTY] * 9
        self.current_player = PLAYER_X
        return "".join(self.board)

    def available_actions(self):
        return [i for i, spot in enumerate(self.board) if spot == EMPTY]

    def step(self, action):
        self.board[action] = self.current_player
        winner = self.check_winner()
        done = winner or not self.available_actions()
        reward = 1 if winner == self.current_player else 0
        self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X
        return "".join(self.board), reward, done

    def check_winner(self, board=None):
        board = board or self.board
        for a, b, c in WINNING_LINES:
            if board[a] == board[b] == board[c] != EMPTY:
                return board[a]

    def render(self):
        print("\n".join([" | ".join(self.board[i:i + 3]) for i in range(0, 9, 3)]) + "\n")


class QLearningAgent:  # Trained agent that plays Tic-Tac-Toe from Q-learning
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q = {}
        self.epsilon, self.alpha, self.gamma = epsilon, alpha, gamma

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def best_action(self, state, actions):
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        return random.choice([a for a, q in zip(actions, q_values) if q == max_q])

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self.best_action(state, actions)

    def update(self, state, action, reward, next_state, next_actions, done):
        old_q = self.get_q(state, action)
        future_q = 0 if done or not next_actions else max(self.get_q(next_state, a) for a in next_actions)
        target = reward - self.gamma * future_q
        self.q[(state, action)] = old_q + self.alpha * (target - old_q)


class MinimaxAgent:  # The perfect player
    def __init__(self, player):
        self.player = player
        self.cache = {}

    def choose_action(self, board, actions):
        board_tuple = tuple(board)
        if board_tuple in self.cache:
            return self.cache[board_tuple]

        is_maximizer = self.player == PLAYER_X
        best_val = -float('inf') if is_maximizer else float('inf')
        compare_op = max if is_maximizer else min
        best_move = None

        for action in actions:
            board[action] = self.player
            val = self.minimax(board, not is_maximizer)
            board[action] = EMPTY
            if compare_op(best_val, val) == val:
                best_val, best_move = val, action
        self.cache[board_tuple] = best_move
        return best_move

    def minimax(self, board, is_maximizing, depth=0):
        winner = TicTacToe().check_winner(board)
        if winner:
            return (10 - depth) if winner == PLAYER_X else (depth - 10)
        if EMPTY not in board:
            return 0
        best_val = -float('inf') if is_maximizing else float('inf')
        player = PLAYER_X if is_maximizing else PLAYER_O
        compare_op = max if is_maximizing else min
        for i in range(9):
            if board[i] == EMPTY:
                board[i] = player
                val = self.minimax(board, not is_maximizing, depth + 1)
                board[i] = EMPTY
                best_val = compare_op(best_val, val)
        return best_val


def train(agent, episodes):
    env = TicTacToe()
    for _ in range(episodes):
        state, done = env.reset(), False
        while not done:
            actions = env.available_actions()
            action = agent.choose_action(state, actions)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, env.available_actions(), done)
            state = next_state


def run_test(trained_agent, optimal_agent, num_games):
    draws = 0
    env = TicTacToe()
    for _ in range(num_games):
        env.reset()
        done = False
        while not done:
            actions = env.available_actions()
            if not actions:
                draws += 1
                break
            if env.current_player == PLAYER_X:
                state_str = "".join(env.board)
                action = trained_agent.best_action(state_str, actions)
            else:  # PLAYER_O
                action = optimal_agent.choose_action(env.board, actions)
            env.step(action)
            if env.check_winner():
                break
    print(f"Draws: {draws}/{num_games} ({(draws / num_games) * 100:.1f}%)")
    return draws / num_games


def play_vs_agent(agent):
    env = TicTacToe()
    print("Starting a new game. You are 'O'.")
    while True:
        env.render()
        winner = env.check_winner()
        actions = env.available_actions()
        if not actions or winner: break
        if env.current_player == PLAYER_X:
            print("Agent's move (X):")
            action = agent.best_action("".join(env.board), actions)
        else:
            action = -1
            while action not in actions:
                try:
                    action = int(input(f"Your move (O) - choose from {actions}: "))
                except (ValueError, IndexError):
                    print("Invalid input.")
        env.step(action)
    winner = env.check_winner()
    print(f"{'You win!' if winner == PLAYER_O else 'Agent wins!' if winner == PLAYER_X else 'It is a draw!'}\n\n")


if __name__ == "__main__":
    INTERACTIVE_MODE = False

    if INTERACTIVE_MODE:
        while True:
            q_agent = QLearningAgent()
            print("Training agent for interactive play...")
            train(q_agent, episodes=25000)
            play_vs_agent(q_agent)
    else:
        results = []
        episode_steps = range(0, 30001, 1500)
        for episodes in episode_steps:
            print(f"\n--- Training with {episodes} episodes ---")
            q_agent = QLearningAgent()
            train(q_agent, episodes=episodes)
            optimal_agent = MinimaxAgent(player=PLAYER_O)
            draw_rate = run_test(q_agent, optimal_agent, num_games=1000)
            results.append(draw_rate)
        print("\n--- Draw Rate vs. Training Episodes ---")
        for episodes, rate in zip(episode_steps, results):
            bar = "#" * int(rate * 50)
            print(f"{episodes:>5} episodes | {rate:.2f} | {bar}")