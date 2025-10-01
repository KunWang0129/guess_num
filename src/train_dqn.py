from dataclasses import dataclass
from typing import Tuple, List

from collections import OrderedDict

import hydra
from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

from src.dqn.agent import Agent
from src.dqn.experience import SequenceReplay, RLDataset, Experience
from src.dqn.mlp import MLP
from src.dqn.embeddingchars import EmbeddingChars
from src.num_env import state
from src.utils.logging_utils import CSVLogger
import src.num_env.wordle as wordle_envs  # Import environment classes

import gymnasium as gym
import os

AVAIL_GPUS = min(1, torch.cuda.device_count())


def construct(network_name: str, obs_size: int, n_actions: int, sequence_list: List[int], hidden_size: int = 256) -> nn.Module:
    """Construct a neural network based on the specified name.

    Args:
        network_name: Name of the network architecture ('MLP', 'EmbeddingChars', etc.)
        obs_size: Observation space size
        n_actions: Number of actions
        sequence_list: List of valid sequences
        hidden_size: Hidden layer size

    Returns:
        Neural network module
    """
    if network_name == 'MLP':
        return MLP(obs_size, n_actions, sequence_list, hidden_size)
    elif network_name == 'EmbeddingChars':
        return EmbeddingChars(obs_size, n_actions, sequence_list, hidden_size)
    else:
        raise ValueError(f"Unknown network type: {network_name}")


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
            self,
            initialize_winning_replays: str = None,
            deep_q_network: str = 'SumChars',
            batch_size: int = 1024,
            lr: float = 1e-2,
            weight_decay: float = 1.e-4,
            env: str = "WordleEnv-v0",
            data_path: str = None,
            gamma: float = 0.9,
            sync_rate: int = 10,
            replay_size: int = 1000,
            hidden_size: int = 256,
            num_workers: int = 0,
            warm_start_size: int = 1000,
            eps_last_frame: int = 10000,
            eps_start: float = 1.0,
            eps_end: float = 0.01,
            episode_length: int = 25,
            warm_start_steps: int = 1000,
            csv_filename: str = 'training_metrics.csv',
            log_interval: int = 100,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()

        self.writer = SummaryWriter()
        self.csv_logger = CSVLogger(
            filename=self.hparams.csv_filename,
            fieldnames=['step', 'loss', 'reward', 'wins', 'losses', 'lose_ratio', 'reward_per_game', 'avg_winning_turns']
        )

        # Instantiate environment class directly with data_path
        env_class = getattr(wordle_envs, self.hparams.env)
        self.env = env_class(data_path=self.hparams.data_path)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self._winning_steps = 0
        self._wins = 0
        self._losses = 0
        self._rewards = 0

        print("dqn:", self.hparams.env, self.env.max_turns, n_actions, obs_size)

        self.net = construct(
            self.hparams.deep_q_network, obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size, sequence_list=self.env.sequences)
        self.target_net = construct(
            self.hparams.deep_q_network, obs_size=obs_size, n_actions=n_actions, hidden_size=hidden_size, sequence_list=self.env.sequences)

        self.dataset = RLDataset(
            winners=SequenceReplay(self.hparams.replay_size//2, self.hparams.initialize_winning_replays),
            losers=SequenceReplay(self.hparams.replay_size//2),
            sample_size=self.hparams.episode_length)

        self.agent = Agent(self.net, self.env.action_space)
        self.state, _ = self.env.reset()
        self.total_reward = 0
        self.episode_reward = 0
        self.total_games_played = 0
        self.populate(self.hparams.warm_start_steps)

    def on_train_start(self) -> None:
        """Called when training starts - set up CSV logger with trainer log directory."""
        if self.trainer and self.trainer.log_dir:
            self.csv_logger.set_log_dir(self.trainer.log_dir)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.play_game(epsilon=1., device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple) -> torch.Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    @torch.no_grad()
    def play_game(
            self,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Tuple[float, bool]:
        done = False
        cur_seq = list()
        reward = 0
        while not done:
            exp = self.play_step(epsilon, device)
            done = exp.done
            reward = exp.reward
            cur_seq.append(exp)

        winning_steps = self.env.max_turns - state.remaining_steps(self.state)
        if reward > 0:
            self.dataset.winners.append(cur_seq)
        else:
            self.dataset.losers.append(cur_seq)
        self.state, _ = self.env.reset()

        return reward, winning_steps

    def play_step(
            self,
            epsilon: float = 0.0,
            device: str = "cpu",
    ) -> Experience:
        action = self.agent.get_action(self.state, epsilon, device)

        # do step in the environment
        new_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        exp = Experience(self.state.copy(), action, reward, done, new_state.copy(), self.env.goal_sequence)

        self.state = new_state
        return exp

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.global_step / self.hparams.eps_last_frame,
            )

        # step through environment with agent
        with torch.no_grad():
            reward, winning_steps = self.play_game(epsilon, device)
        self.total_games_played += 1
        if reward > 0:
            self._wins += 1
            self._winning_steps += winning_steps
        else:
            self._losses += 1

        self._rewards += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss.detach(),
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        if self.global_step % self.hparams.log_interval == 0:
            if len(self.dataset.winners) > 0:
                winner = self.dataset.winners.buffer[-1]
                game = f"goal: {self.env.sequences[winner[0].goal_id]}\n"
                for i, xp in enumerate(winner):
                    game += f"{i}: {self.env.sequences[xp.action]}\n"
                self.writer.add_text("game sample/winner", game, global_step=self.global_step)
            if len(self.dataset.losers) > 0:
                loser = self.dataset.losers.buffer[-1]
                game = f"goal: {self.env.sequences[loser[0].goal_id]}\n"
                for i, xp in enumerate(loser):
                    game += f"{i}: {self.env.sequences[xp.action]}\n"
                self.writer.add_text("game sample/loser", game, global_step=self.global_step)

            # TensorBoard logging
            self.writer.add_scalar("train_loss", loss, global_step=self.global_step)
            self.writer.add_scalar("total_games_played", self.total_games_played, global_step=self.global_step)
            self.writer.add_scalar("winner_buffer", len(self.dataset.winners), global_step=self.global_step)
            self.writer.add_scalar("loser_buffer", len(self.dataset.losers), global_step=self.global_step)
            self.writer.add_scalar("lose_ratio", self._losses/(self._wins+self._losses), global_step=self.global_step)
            self.writer.add_scalar("wins", self._wins, global_step=self.global_step)
            self.writer.add_scalar("reward_per_game", self._rewards / (self._wins+self._losses), global_step=self.global_step)

            # CSV logging
            csv_metrics = {
                'step': self.global_step,
                'loss': loss.item(),
                'reward': self._rewards / (self._wins+self._losses),
                'wins': self._wins,
                'losses': self._losses,
                'lose_ratio': self._losses/(self._wins+self._losses),
                'reward_per_game': self._rewards / (self._wins+self._losses),
            }

            if self._wins > 0:
                avg_winning_turns = self._winning_steps/self._wins
                self.writer.add_scalar("avg_winning_turns", avg_winning_turns, global_step=self.global_step)
                csv_metrics['avg_winning_turns'] = avg_winning_turns
            else:
                csv_metrics['avg_winning_turns'] = 0

            self.csv_logger.log(csv_metrics)

            self._winning_steps = 0
            self._wins = 0
            self._losses = 0
            self._rewards = 0

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return str(batch[0].device)


@hydra.main(version_base=None, config_path="../conf", config_name="train_dqn")
def main(cfg: DictConfig):
    """Main training function using Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Build data path from config
    data_filename = f"{cfg.data.output.prefix}_{cfg.data.family.name}.{cfg.data.output.format}"
    data_path = os.path.join(cfg.data.output.dir, data_filename)

    model = DQNLightning(
        initialize_winning_replays=cfg.train.initialize_winning_replays,
        deep_q_network=cfg.network.name,
        env=cfg.env.env_class,
        data_path=data_path,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        replay_size=cfg.train.replay_size,
        batch_size=cfg.train.batch_size,
        sync_rate=cfg.train.sync_rate,
        episode_length=cfg.train.episode_length,
        hidden_size=cfg.network.hidden_size,
        num_workers=cfg.train.num_workers,
        eps_start=cfg.train.eps_start,
        eps_end=cfg.train.eps_end,
        eps_last_frame=int(cfg.train.max_epochs * cfg.train.last_frame_cutoff),
        csv_filename=cfg.train.logging.csv_filename,
        log_interval=cfg.train.logging.log_interval,
    )

    @dataclass
    class SaveBufferCallback(Callback):
        buffer: SequenceReplay
        filename: str

        def on_train_end(self, trainer, pl_module):
            path = f'{trainer.log_dir}/checkpoints'
            fname = self.filename,
            self.buffer.save(f'{path}/{fname}')

    save_buffer_callback = SaveBufferCallback(buffer=model.dataset.winners, filename='sequence_buffer.pkl')
    model_checkpoint = ModelCheckpoint(every_n_epochs=cfg.train.checkpoint_every_n_epochs)

    trainer = Trainer(
        accelerator="auto",
        devices=AVAIL_GPUS if AVAIL_GPUS > 0 else "auto",
        max_epochs=cfg.train.max_epochs,
        enable_checkpointing=True,
        callbacks=[model_checkpoint, save_buffer_callback],
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
