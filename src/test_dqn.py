from typing import Dict
import csv

import hydra
from omegaconf import DictConfig
import torch
import gymnasium as gym

from src.train_dqn import DQNLightning
from src.dqn.agent import Agent
from src.num_env import state
import src.num_env.wordle  # Import to register environments


def evaluate_model(
    model: DQNLightning,
    num_episodes: int,
    device: str = "cpu"
) -> Dict[str, float]:
    """Evaluate the DQN model on multiple episodes.

    Args:
        model: Trained DQN Lightning model
        num_episodes: Number of episodes to evaluate
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    model.to(device)

    env = gym.make(model.hparams.env, data_path=model.hparams.data_path)
    agent = Agent(model.net, env.action_space)

    total_wins = 0
    total_losses = 0
    total_winning_turns = 0
    total_rewards = 0

    with torch.no_grad():
        for episode in range(num_episodes):
            state_obs, _ = env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Greedy action selection (epsilon=0)
                action = agent.get_action(state_obs, epsilon=0.0, device=device)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state_obs = next_state

            total_rewards += episode_reward

            if episode_reward > 0:
                total_wins += 1
                winning_turns = env.unwrapped.max_turns - state.remaining_steps(state_obs)
                total_winning_turns += winning_turns
            else:
                total_losses += 1

            if (episode + 1) % 100 == 0:
                print(f"Evaluated {episode + 1}/{num_episodes} episodes")

    total_games = total_wins + total_losses
    metrics = {
        'total_episodes': num_episodes,
        'wins': total_wins,
        'losses': total_losses,
        'win_rate': total_wins / total_games if total_games > 0 else 0,
        'loss_rate': total_losses / total_games if total_games > 0 else 0,
        'avg_reward': total_rewards / total_games if total_games > 0 else 0,
        'avg_winning_turns': total_winning_turns / total_wins if total_wins > 0 else 0,
    }

    return metrics


def save_metrics_to_csv(metrics: Dict[str, float], output_path: str):
    """Save evaluation metrics to CSV file.

    Args:
        metrics: Dictionary of metrics to save
        output_path: Path to output CSV file
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics saved to {output_path}")


@hydra.main(version_base=None, config_path="../conf", config_name="test_dqn")
def main(cfg: DictConfig):
    """Main testing function using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    print(f"Loading checkpoint from: {cfg.test.checkpoint_path}")

    # Load model from checkpoint
    model = DQNLightning.load_from_checkpoint(cfg.test.checkpoint_path)

    device = "cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu"
    print(f"Using device: {device}")

    print(f"Evaluating on {cfg.test.num_episodes} episodes...")
    metrics = evaluate_model(
        model=model,
        num_episodes=cfg.test.num_episodes,
        device=device
    )

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Total Episodes: {metrics['total_episodes']}")
    print(f"Wins: {metrics['wins']}")
    print(f"Losses: {metrics['losses']}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Loss Rate: {metrics['loss_rate']:.2%}")
    print(f"Average Reward: {metrics['avg_reward']:.3f}")
    if metrics['avg_winning_turns'] > 0:
        print(f"Average Winning Turns: {metrics['avg_winning_turns']:.2f}")

    # Save to CSV if output path specified
    if cfg.test.output_csv:
        save_metrics_to_csv(metrics, cfg.test.output_csv)


if __name__ == '__main__':
    main()
