import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from environment import TradingEnv
from model import DQN
from replay_buffer import ReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DQN trading agent")
    parser.add_argument("--data", type=Path, required=True, help="Path to OHLCV CSV data")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=int, default=500)
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--window_size", type=int, default=10)
    return parser.parse_args()


def load_data(path: Path):
    df = pd.read_csv(path)
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required_cols}")
    return df


def select_action(state, policy_net, epsilon, action_space):
    if np.random.rand() < epsilon:
        return action_space.sample()
    with torch.no_grad():
        state_v = torch.from_numpy(state).unsqueeze(0)
        q_values = policy_net(state_v)
        return int(torch.argmax(q_values, dim=1).item())


def compute_td_target(batch, policy_net, target_net, gamma):
    state_batch = torch.tensor(np.array(batch.state))
    action_batch = torch.tensor(batch.action)
    reward_batch = torch.tensor(batch.reward)
    next_state_batch = torch.tensor(np.array(batch.next_state))
    done_batch = torch.tensor(batch.done, dtype=torch.float32)

    q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
    with torch.no_grad():
        next_q = target_net(next_state_batch).max(1).values
    target = reward_batch + gamma * next_q * (1 - done_batch)
    loss = nn.functional.mse_loss(q_values, target)
    return loss


def main():
    args = parse_args()
    df = load_data(args.data)

    env = TradingEnv(df, window_size=args.window_size)
    eval_env = TradingEnv(df, window_size=args.window_size)

    obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    n_actions = env.action_space.n

    policy_net = DQN(obs_dim, n_actions)
    target_net = DQN(obs_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size)

    steps_done = 0
    epsilon = args.epsilon_start

    for ep in range(args.episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = select_action(state, policy_net, epsilon, env.action_space)
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            steps_done += 1
            epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
                np.exp(-1.0 * steps_done / args.epsilon_decay)

            if len(replay_buffer) >= args.batch_size:
                batch = replay_buffer.sample(args.batch_size)
                loss = compute_td_target(batch, policy_net, target_net, args.gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ep % args.target_update == 0 and steps_done > 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {ep}: reward={episode_reward:.2f}")

    # Simple evaluation
    state, _ = eval_env.reset()
    done = False
    total_reward = 0.0
    while not done:
        with torch.no_grad():
            state_v = torch.from_numpy(state).unsqueeze(0)
            action = int(torch.argmax(policy_net(state_v)))
        next_state, reward, done, _, _ = eval_env.step(action)
        total_reward += reward
        state = next_state
    print(f"Evaluation total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
