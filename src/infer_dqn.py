import hydra
from omegaconf import DictConfig
import torch
import gymnasium as gym

from src.train_dqn import DQNLightning
from src.dqn.agent import Agent
from src.num_env import state
import src.num_env.wordle  # Import to register environments


def infer_single_sequence(
    model: DQNLightning,
    target_sequence: str,
    show_steps: bool = True,
    device: str = "cpu"
):
    """Run inference on a single target sequence.

    Args:
        model: Trained DQN Lightning model
        target_sequence: Target sequence to guess
        show_steps: Whether to print step-by-step progress
        device: Device to run inference on

    Returns:
        Tuple of (success: bool, num_turns: int)
    """
    model.eval()
    model.to(device)

    env = gym.make(model.hparams.env, data_path=model.hparams.data_path)
    agent = Agent(model.net, env.action_space)

    # Verify target sequence is in vocabulary
    if target_sequence not in env.unwrapped.sequences:
        raise ValueError(f"Target sequence '{target_sequence}' not found in environment vocabulary")

    # Set the goal sequence
    env.unwrapped.set_goal_sequence(target_sequence)

    state_obs, _ = env.reset()
    # Re-set goal after reset (since reset randomizes it)
    env.unwrapped.set_goal_sequence(target_sequence)

    if show_steps:
        print(f"\n=== Guessing sequence: {target_sequence} ===\n")

    done = False
    turn = 0
    final_reward = 0

    with torch.no_grad():
        while not done:
            # Greedy action selection (epsilon=0)
            action = agent.get_action(state_obs, epsilon=0.0, device=device)
            guessed_sequence = env.unwrapped.sequences[action]

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            turn += 1

            if show_steps:
                if terminated:
                    print(f"Turn {turn}: {guessed_sequence} ✓ (CORRECT!)")
                else:
                    remaining = state.remaining_steps(next_state)
                    print(f"Turn {turn}: {guessed_sequence} (remaining turns: {remaining})")

            state_obs = next_state
            final_reward = reward

    success = final_reward > 0

    if show_steps:
        print(f"\n=== Result ===")
        if success:
            print(f"✓ SUCCESS! Found sequence in {turn} turns")
        else:
            print(f"✗ FAILED! Could not find sequence in {env.unwrapped.max_turns} turns")

    return success, turn


@hydra.main(version_base=None, config_path="../conf", config_name="infer_dqn")
def main(cfg: DictConfig):
    """Main inference function using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    print(f"Loading checkpoint from: {cfg.infer.checkpoint_path}")

    # Load model from checkpoint
    model = DQNLightning.load_from_checkpoint(cfg.infer.checkpoint_path)

    device = "cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu"
    print(f"Using device: {device}")

    # Run inference
    success, num_turns = infer_single_sequence(
        model=model,
        target_sequence=cfg.infer.target_sequence,
        show_steps=cfg.infer.show_steps,
        device=device
    )

    return success, num_turns


if __name__ == '__main__':
    main()
