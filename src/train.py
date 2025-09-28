"""
Training Script for Sequence Guessing RL Agent with Options Framework

This is the main entry point for training the DQN agent with hierarchical options.
Uses Hydra for configuration management and PyTorch Lightning for training structure.
"""

import os
import logging
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra
from omegaconf import DictConfig, OmegaConf
import glob

from .environment import SequenceGuessingEnv, SequenceGuessingConfig
from .agent import DQNAgent, AgentConfig
from .model import ModelConfig
from .data import create_sequence_loader

# Set up logging
log = logging.getLogger(__name__)


def setup_logging(config: DictConfig) -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.logging.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )


def cleanup_old_checkpoints(checkpoint_dir: str, pattern: str, keep_latest: int = 1) -> None:
    """
    Remove old checkpoint files, keeping only the latest ones

    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern to match checkpoint files
        keep_latest: Number of latest checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return

    checkpoint_pattern = os.path.join(checkpoint_dir, pattern)
    checkpoint_files = glob.glob(checkpoint_pattern)

    if len(checkpoint_files) <= keep_latest:
        return

    # Sort by modification time (newest first)
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)

    # Remove old checkpoints
    for old_checkpoint in checkpoint_files[keep_latest:]:
        try:
            os.remove(old_checkpoint)
            log.info(f"Removed old checkpoint: {old_checkpoint}")
        except OSError as e:
            log.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")


def create_environment(config: DictConfig) -> SequenceGuessingEnv:
    """
    Create environment from configuration

    Args:
        config: Hydra configuration

    Returns:
        Initialized environment
    """
    env_config = SequenceGuessingConfig(
        sequence_length=config.env.sequence_length,
        max_episode_steps=config.env.max_episode_steps,
        num_values=config.env.num_values,
        reward_correct_guess=config.env.reward_correct_guess,
        reward_wrong_guess=config.env.reward_wrong_guess,
        reward_step=config.env.reward_step,
        reward_partial_correct=config.env.reward_partial_correct,
        provide_feedback=config.env.provide_feedback,
        allow_repeated_guesses=config.env.allow_repeated_guesses,
        verbose=config.env.verbose
    )

    sequence_provider = None
    data_loader_cfg = config.get("data_loader")
    if data_loader_cfg and data_loader_cfg.get("enabled", False):
        sequence_provider = create_sequence_loader(
            data_loader_cfg,
            sequence_length=env_config.sequence_length,
        )
        dataset_summary = sequence_provider.dataset.summary()
        log.info(
            "Loaded sequence dataset: %s (size=%d, families=%s)",
            dataset_summary.get("path"),
            dataset_summary.get("size"),
            dataset_summary.get("families"),
        )

    return SequenceGuessingEnv(env_config, sequence_provider=sequence_provider)


def create_agent_config(config: DictConfig) -> AgentConfig:
    """
    Create agent configuration from Hydra config

    Args:
        config: Hydra configuration

    Returns:
        Agent configuration
    """
    agent_cfg = config.agent
    macro_flag = agent_cfg.get("use_macro_options")
    if macro_flag is None:
        legacy_flag = agent_cfg.get("use_options")
        if legacy_flag is not None:
            macro_flag = legacy_flag
    if macro_flag is None:
        macro_flag = True

    return AgentConfig(
        # Learning parameters
        learning_rate=agent_cfg.learning_rate,
        gamma=agent_cfg.gamma,
        batch_size=agent_cfg.batch_size,

        # Exploration parameters
        eps_start=agent_cfg.eps_start,
        eps_end=agent_cfg.eps_end,
        eps_decay=agent_cfg.eps_decay,

        # Target network update
        target_update_freq=agent_cfg.target_update_freq,
        soft_update_tau=agent_cfg.soft_update_tau,
        use_soft_updates=agent_cfg.use_soft_updates,

        # Training parameters
        max_episodes=agent_cfg.max_episodes,
        max_steps_per_episode=agent_cfg.max_steps_per_episode,

        # Options-specific parameters
        use_macro_options=macro_flag,
        option_epsilon=agent_cfg.option_epsilon,
        option_termination_bonus=agent_cfg.option_termination_bonus,

        # Network architecture
        model_type=agent_cfg.model_type,
        hidden_sizes=agent_cfg.hidden_sizes,
        activation=agent_cfg.activation,
        dropout_rate=agent_cfg.dropout_rate,

        # Replay buffer
        buffer_size=agent_cfg.buffer_size,
        min_replay_size=agent_cfg.min_replay_size,
        prioritized_replay=agent_cfg.prioritized_replay,

        # Options configuration
        options=getattr(agent_cfg, 'options', None)
    )


def setup_callbacks(config: DictConfig) -> list:
    """Setup PyTorch Lightning callbacks"""
    callbacks = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.experiment.checkpoint_dir,
        filename='dqn-{epoch:02d}-{episode_reward:.2f}',
        monitor='episode_reward',
        mode='max',
        save_top_k=3,
        save_last=True,
        every_n_epochs=config.training.save_checkpoint_every
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config.training.early_stopping_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor='episode_reward',
            patience=config.training.early_stopping_patience,
            mode='max',
            min_delta=0.01
        )
        callbacks.append(early_stop_callback)

    return callbacks


def setup_logger(config: DictConfig) -> pl.loggers.Logger:
    """Setup PyTorch Lightning logger"""
    if config.logging.tensorboard:
        return TensorBoardLogger(
            save_dir=config.logging.log_dir,
            name=config.experiment.name,
            version=None
        )
    else:
        return None


def train_agent(config: DictConfig) -> DQNAgent:
    """
    Main training function

    Args:
        config: Hydra configuration

    Returns:
        Trained agent
    """
    # Set random seeds for reproducibility
    pl.seed_everything(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create environment
    env = create_environment(config)
    log.info(f"Created environment with config: {env.get_info()}")

    # Create agent
    agent_config = create_agent_config(config)
    agent = DQNAgent(env, agent_config)
    log.info(f"Created agent with {sum(p.numel() for p in agent.parameters())} parameters")

    # Log configuration
    log.info(f"Training configuration:\n{OmegaConf.to_yaml(config)}")

    # Training loop (manual since we need environment interaction)
    log.info("Starting training...")

    best_reward = float('-inf')
    episodes_without_improvement = 0

    for episode in range(config.training.max_episodes):
        # Run episode
        episode_stats = agent.run_episode()

        # Log episode statistics
        if episode % 10 == 0:
            log.info(
                f"Episode {episode}: "
                f"Reward={episode_stats['episode_reward']:.2f}, "
                f"Length={episode_stats['episode_length']}, "
                f"Epsilon={agent.epsilon:.3f}, "
                f"Buffer={len(agent.replay_buffer)}"
            )

        # Log option usage
        if agent.options_manager and episode % 50 == 0:
            option_stats = agent.options_manager.get_option_stats()
            log.info(f"Option usage: {option_stats}")

        # Evaluation
        if episode % config.evaluation.eval_every == 0 and episode > 0:
            eval_stats = agent.evaluate(config.evaluation.eval_episodes)
            log.info(
                f"Evaluation after episode {episode}: "
                f"Mean reward={eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}, "
                f"Success rate={eval_stats['success_rate']:.2%}"
            )

            # Check for improvement
            current_reward = eval_stats['mean_reward']
            if current_reward > best_reward:
                best_reward = current_reward
                episodes_without_improvement = 0

                # Save best model
                if config.evaluation.save_best_model:
                    checkpoint_path = os.path.join(
                        config.experiment.checkpoint_dir,
                        f"best_model_episode_{episode}.ckpt"
                    )
                    agent.save_checkpoint(checkpoint_path)
                    log.info(f"Saved best model with reward {best_reward:.2f}")

                    # Keep only the latest best model
                    cleanup_old_checkpoints(
                        config.experiment.checkpoint_dir,
                        "best_model_episode_*.ckpt",
                        keep_latest=1
                    )

            else:
                episodes_without_improvement += config.evaluation.eval_every

            # Early stopping
            if episodes_without_improvement >= config.training.early_stopping_patience:
                log.info(f"Early stopping after {episode} episodes without improvement")
                break

        # Save periodic checkpoints
        if episode % config.training.save_checkpoint_every == 0 and episode > 0:
            checkpoint_path = os.path.join(
                config.experiment.checkpoint_dir,
                f"checkpoint_episode_{episode}.ckpt"
            )
            agent.save_checkpoint(checkpoint_path)

            # Keep only the latest periodic checkpoint
            cleanup_old_checkpoints(
                config.experiment.checkpoint_dir,
                "checkpoint_episode_*.ckpt",
                keep_latest=1
            )

    # Final evaluation
    log.info("Training completed. Running final evaluation...")
    final_stats = agent.evaluate(config.evaluation.eval_episodes * 2)
    log.info(f"Final evaluation: {final_stats}")

    # Save final model
    final_checkpoint_path = os.path.join(
        config.experiment.checkpoint_dir,
        "final_model.ckpt"
    )
    agent.save_checkpoint(final_checkpoint_path)
    log.info(f"Saved final model to {final_checkpoint_path}")

    return agent


def run_experiment(config: DictConfig) -> None:
    """
    Run complete experiment with the given configuration

    Args:
        config: Hydra configuration
    """
    # Create directories
    os.makedirs(config.experiment.save_dir, exist_ok=True)
    os.makedirs(config.experiment.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)

    # Clean up any existing old checkpoints at start
    cleanup_old_checkpoints(config.experiment.checkpoint_dir, "best_model_episode_*.ckpt", keep_latest=1)
    cleanup_old_checkpoints(config.experiment.checkpoint_dir, "checkpoint_episode_*.ckpt", keep_latest=1)

    # Setup logging
    setup_logging(config)
    log.info("Starting experiment...")

    try:
        # Train agent
        agent = train_agent(config)

        # Get final training statistics
        training_stats = agent.get_training_stats()
        log.info(f"Training statistics: {training_stats}")

        log.info("Experiment completed successfully!")

    except Exception as e:
        log.error(f"Experiment failed with error: {str(e)}")
        raise


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main entry point for training script

    Args:
        config: Hydra configuration loaded from conf/ directory
    """
    print("=" * 80)
    print("Sequence Guessing RL Agent with Options Framework")
    print("=" * 80)

    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()


def create_demo_script():
    """
    Create a simple demo script that can be run without full configuration
    """
    def demo():
        """Minimal demo of the system working"""
        print("Running minimal demo...")

        # Create simple environment
        env_config = SequenceGuessingConfig(
            sequence_length=3,
            max_episode_steps=50,
            verbose=True
        )
        env = SequenceGuessingEnv(env_config)

        # Create simple agent config
        agent_config = AgentConfig(
            max_episodes=5,
            max_steps_per_episode=20,
            learning_rate=0.01,
            eps_start=0.8,
            eps_end=0.1,
            buffer_size=1000,
            min_replay_size=100,
            use_macro_options=True
        )

        # Create agent
        agent = DQNAgent(env, agent_config)

        print(f"Environment info: {env.get_info()}")
        print(f"Agent total actions: {agent.total_actions}")
        print(f"Available options: {list(agent.options_manager.options.keys())}")

        # Run a few episodes
        for episode in range(3):
            stats = agent.run_episode()
            print(f"Episode {episode}: Reward={stats['episode_reward']:.2f}, "
                  f"Length={stats['episode_length']}")

        print("Demo completed!")

    return demo


# Create demo function for easy testing
demo = create_demo_script()
