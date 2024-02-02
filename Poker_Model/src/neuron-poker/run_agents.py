from poker_agent_sarsa import Sarsa_PokerAgent
from poker_agent_q_learning import QLearning_PokerAgent
from poker_agent_expected_sarsa import ExpectedSarsa_PokerAgent
# from poker_agent_dqn import DQN_PokerAgent, MethodToUse
from gym_env.env import HoldemTable
from gym_env.env import PlayerShell
from agents.agent_random import Player as RandomPlayer

def run_sarsa_agent():
    env = HoldemTable()
    env.add_player(RandomPlayer())
    env.add_player(PlayerShell(name='SARSA', stack_size=500))
    env.reset()
    env.seed(42)
    sarsaAgent = Sarsa_PokerAgent(env, gamma=0.8, alpha=1e-1,
                        start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
    sarsaAgent.train(no_episodes=500)

def run_qlearning_agent():
    env = HoldemTable()
    env.add_player(RandomPlayer())
    env.add_player(PlayerShell(name='Q_Learning', stack_size=500))
    env.reset()
    env.seed(42)

    QLearningAgent = QLearning_PokerAgent(env, gamma=0.8, alpha=1e-1,
                        start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
    QLearningAgent.train(no_episodes=500)

def run_expected_sarsa_agent():
    env = HoldemTable()
    env.add_player(RandomPlayer())
    env.add_player(PlayerShell(name='Expected_SARSA', stack_size=500))
    env.reset()
    env.seed(42)

    expectedSarsaAgent = ExpectedSarsa_PokerAgent(env, gamma=0.8, alpha=1e-1,
                            start_epsilon=1, end_epsilon=1e-2, epsilon_decay=0.999)
    expectedSarsaAgent.train(no_episodes=500)
