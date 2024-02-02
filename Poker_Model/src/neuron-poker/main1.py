import sys
import pandas as pd
import numpy as np

from gym_env.env import HoldemTable
from gym_env.env import PlayerShell
from tools.helper import get_config

from agents.agent_random import Player as RandomPlayer
from poker_agent_expected_sarsa import ExpectedSarsa_PokerAgent
from poker_agent_q_learning import QLearning_PokerAgent
from poker_agent_sarsa import Sarsa_PokerAgent


"""
Usage:
  main.py [option]

option:
    sarsa                   -- 5 random players + 1 SARSA player
    expected_sarsa          -- 5 random players + 1 Expected SARSA player
    q_learning              -- 5 random players + 1 Q Learning player
    all_basic               -- 1 SARSA player + 1 Expected SARSA player + 1 Q Learning player
"""


class SelfPlay:
    def __init__(self, render, num_episodes, use_cpp_montecarlo, funds_plot, stack=500):
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.num_episodes = num_episodes
        self.stack = stack
        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)

    def sarsa_agent(self):
        """An environment with 5 random players and a sarsa player"""
        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        for _ in range(5):
            player = RandomPlayer()
            self.env.add_player(player)
        self.env.add_player(PlayerShell(name='SARSA', stack_size=self.stack))

        self.env.reset()

        sarsaAgent = Sarsa_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                        start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)

        scores = []
        actions_per_ep = []

        for _ in range(self.num_episodes):
            self.env.reset()

            score, actions = sarsaAgent.play(no_episodes=1)
            scores.append(score)
            actions_per_ep.append(actions)

            self.winner_in_episodes.append(self.env.winner_ix)

        average_score = np.mean(scores)
        average_actions = np.mean(actions_per_ep)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")
        print('------------')
        print(f'Average score: {average_score}, Average actions per ep: {average_actions}')

    def expected_sarsa_agent(self):
        """Create an environment with 5 random players and a expected sarsa player"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        for _ in range(5):
            player = RandomPlayer()
            self.env.add_player(player)
        self.env.add_player(PlayerShell(name='Expected_SARSA', stack_size=self.stack))

        self.env.reset()
        
        expectedSarsaAgent = ExpectedSarsa_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)

        scores = []
        actions_per_ep = []

        for _ in range(self.num_episodes):
            self.env.reset()

            score, actions = expectedSarsaAgent.play(no_episodes=1)
            scores.append(score)
            actions_per_ep.append(actions)

            self.winner_in_episodes.append(self.env.winner_ix)

        average_score = np.mean(scores)
        average_actions = np.mean(actions_per_ep)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")
        print('------------')
        print(f'Average score: {average_score}, Average actions per ep: {average_actions}')

    def q_learning_agent(self):
        """Create an environment with 5 random players and a q learning player"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        for _ in range(5):
            player = RandomPlayer()
            self.env.add_player(player)
        self.env.add_player(PlayerShell(name='Q_Learning', stack_size=self.stack))

        self.env.reset()
        
        QLearningAgent = QLearning_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)

        scores = []
        actions_per_ep = []

        for _ in range(self.num_episodes):
            self.env.reset()

            score, actions = QLearningAgent.play(no_episodes=1)
            scores.append(score)
            actions_per_ep.append(actions)

            self.winner_in_episodes.append(self.env.winner_ix)

        average_score = np.mean(scores)
        average_actions = np.mean(actions_per_ep)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")
        print('------------')
        print(f'Average score: {average_score}, Average actions per ep: {average_actions}')

    def all_basic_agents(self):
        """Create an environment with all 3 basic players"""

        self.env = HoldemTable(initial_stacks=self.stack, render=self.render)
        self.env.add_player(PlayerShell(name='SARSA', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='Expected_SARSA', stack_size=self.stack))
        self.env.add_player(PlayerShell(name='Q_Learning', stack_size=self.stack))

        self.env.reset()

        sarsaAgent = Sarsa_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                    start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
        sarsaAgent.load_q_table()

        expectedSarsaAgent = ExpectedSarsa_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
        expectedSarsaAgent.load_q_table()

        QLearningAgent = QLearning_PokerAgent(self.env, gamma=0.8, alpha=1e-1,
                                start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
        QLearningAgent.load_q_table()

        for _ in range(self.num_episodes):
            self.env.reset()

            sarsaAgent.play(no_episodes=1)

            self.winner_in_episodes.append(self.env.winner_ix)


        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print("Player - No episodes won")
        print(league_table)
        print(f"Best Player: {best_player}")

def command_line_parser():
    args = sys.argv[1]
    _ = get_config()

    num_episodes = 3
    runner = SelfPlay(render=True, num_episodes=num_episodes, use_cpp_montecarlo=False,
                      funds_plot=True, stack=20)

    if args == 'sarsa':
        runner.sarsa_agent()

    elif args == 'expected_sarsa':
        runner.expected_sarsa_agent()

    elif args == 'q_learning':
        runner.q_learning_agent()
    
    elif args == 'all_basic':
        runner.all_basic_agents()

    else:
        raise RuntimeError("Argument either of the argument to implement it - \nsarsa\nexpected_sarsa\nq_learning\nall_basic\n")


if __name__ == '__main__':
    command_line_parser()