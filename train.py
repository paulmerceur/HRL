import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Tuple

from agent import Agent
from four_rooms_env import make_env_for_vectorization


def train():
    # Hyperparameters - balanced approach for sparse reward learning with RND
    learning_rate = 5e-4  # Slightly higher for faster learning
    rnd_learning_rate = 1e-3  # Learning rate for RND predictor
    num_workers = 12
    num_envs_per_worker = 4
    num_envs = num_workers * num_envs_per_worker
    num_steps = 512
    total_timesteps = 1000000
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 4
    update_epochs = 4  # Restored for more stable learning
    clip_coef = 0.2
    ent_coef = 0.01  # Reduced since RND provides exploration
    vf_coef = 0.5
    max_grad_norm = 0.5
    
    # RND hyperparameters
    intrinsic_reward_coef = 1.0  # Scaling factor for intrinsic rewards
    rnd_update_proportion = 0.25  # Update RND on this proportion of data

    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_updates = total_timesteps // batch_size

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Use standard Gymnasium vectorization
    envs = gym.vector.SyncVectorEnv([make_env_for_vectorization for _ in range(num_envs)])

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs.single_observation_space, envs.single_action_space).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    
    # Separate optimizer for RND predictor network
    rnd_optimizer = optim.Adam(agent.rnd.predictor_network.parameters(), lr=rnd_learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_shape = envs.single_observation_space.shape
    assert obs_shape is not None, "observation space shape cannot be None"
    # For Discrete action spaces, shape is None, so we use empty tuple
    action_shape = envs.single_action_space.shape or ()
    
    obs = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + action_shape, dtype=torch.long).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    intrinsic_rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=42)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_envs).to(device)
    
    # Initialize LSTM state
    lstm_state = (
        torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, num_envs, agent.lstm.hidden_size).to(device),
    )

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if True:  # Could add learning rate annealing here
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        episodes_completed_this_update = 0

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, lstm_state = agent.get_action_and_value(
                    next_obs.unsqueeze(0), lstm_state, None
                )
                values[step] = value.flatten()
                
                # Get intrinsic reward from RND
                intrinsic_reward = agent.get_intrinsic_reward(next_obs.unsqueeze(0))
                intrinsic_rewards[step] = intrinsic_reward.flatten()
                
            actions[step] = action.flatten()
            logprobs[step] = logprob.flatten()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.squeeze(0).cpu().numpy())
            
            rewards[step] = torch.tensor(reward, dtype=torch.float32).to(device).view(-1)

            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                np.logical_or(terminated, truncated)
            ).to(device)

            # Log successful episodes
            for i in range(len(terminated)):
                if terminated[i] or truncated[i]:
                    episodes_completed_this_update += 1
                    if terminated[i] and reward[i] > 0:
                        #print(f"🎉 SUCCESS! Agent got reward {reward[i]:.3f} in step {step} (env={i})")
                        pass

        # Combine extrinsic and intrinsic rewards
        combined_rewards = rewards + intrinsic_reward_coef * intrinsic_rewards

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_action_and_value(
                next_obs.unsqueeze(0), lstm_state, None
            )[3].reshape(1, -1)
            advantages = torch.zeros_like(combined_rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = combined_rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # LSTM forward pass - FIXED: Remove extra dimension from actions
                mb_obs_seq = b_obs[mb_inds].view(minibatch_size // num_envs, num_envs, *obs_shape)
                mb_actions_seq = b_actions.long()[mb_inds].view(minibatch_size // num_envs, num_envs)
                
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    mb_obs_seq, None, mb_actions_seq
                )
                logratio = newlogprob.flatten() - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if True:  # Advantage normalization
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss with stabilization
                newvalue = newvalue.view(-1)
                clamped_returns = torch.clamp(b_returns[mb_inds], -10, 10)
                
                v_loss_unclipped = (newvalue - clamped_returns) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - clamped_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
                v_loss = torch.clamp(v_loss, 0, 100)  # Prevent explosive value loss

                entropy_loss = entropy.flatten().mean()

                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # Update RND predictor network
        rnd_batch_size = int(rnd_update_proportion * batch_size)
        rnd_inds = np.random.choice(batch_size, rnd_batch_size, replace=False)
        rnd_obs = b_obs[rnd_inds].view(rnd_batch_size // num_envs, num_envs, *obs_shape)
        
        rnd_loss = agent.update_rnd(rnd_obs)
        rnd_optimizer.zero_grad()
        rnd_loss.backward()
        rnd_optimizer.step()

        # Logging
        total_nonzero_rewards = torch.sum(rewards > 0).item()
        mean_intrinsic_reward = intrinsic_rewards.mean().item()
        print(f"Update {update}/{num_updates}, "
              f"Episodes Completed: {episodes_completed_this_update}, "
              f"Mean Ext Reward: {rewards.mean().item():.4f}, "
              f"Mean Int Reward: {mean_intrinsic_reward:.4f}, "
              f"Nonzero Rewards: {total_nonzero_rewards}/{num_envs}, "
              f"Reward Range: [{rewards.min().item():.4f}, {rewards.max().item():.4f}], "
              f"Policy Loss: {pg_loss.item():.4f}, "
              f"Value Loss: {v_loss.item():.4f}, "
              f"RND Loss: {rnd_loss.item():.4f}")

        if update % 10 == 0:
            torch.save(agent.state_dict(), 'agent.pth')
            print(f"Saved agent.pth")

    torch.save(agent.state_dict(), 'agent.pth')
    envs.close()

def test():
    """Test the trained agent with a rendered environment."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    env = make_env_for_vectorization(render_mode="human")
    agent = Agent(env.observation_space, env.action_space).to(device)
    
    try:
        agent.load_state_dict(torch.load('agent.pth', map_location=device))
        print("Loaded trained agent")
    except FileNotFoundError:
        print("No trained agent found. Using random agent.")
    
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    lstm_state = (
        torch.zeros(1, 1, agent.lstm_hidden_size).to(device),
        torch.zeros(1, 1, agent.lstm_hidden_size).to(device)
    )
    
    while True:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            action, _, _, _, lstm_state = agent.get_action_and_value(obs_tensor, lstm_state, None)
            
        obs, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += float(reward)
        steps += 1
        
        if terminated or truncated:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'test', 'train_single'])
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test() 