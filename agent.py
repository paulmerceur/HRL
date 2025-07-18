import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class RNDNetwork(nn.Module):
    """Random Network Distillation for intrinsic motivation."""
    def __init__(self, input_size, output_size=64):
        super().__init__()
        self.target_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        self.predictor_network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        target_features = self.target_network(x)
        predicted_features = self.predictor_network(x)
        return target_features, predicted_features
    
    def intrinsic_reward(self, x):
        """Compute intrinsic reward as prediction error."""
        with torch.no_grad():
            target_features, predicted_features = self.forward(x)
            intrinsic_reward = torch.mean((target_features - predicted_features) ** 2, dim=-1)
            return intrinsic_reward


class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        # Handle Dict observation space
        if hasattr(observation_space, 'spaces'):
            # Dict observation space
            self.image_shape = observation_space.spaces['image'].shape  # (H, W, 3)
            self.direction_dim = observation_space.spaces['direction'].n  # 4 directions
        else:
            # Legacy: Box observation space (fallback)
            self.image_shape = observation_space.shape
            self.direction_dim = 4
            
        self.lstm_hidden_size = 128
        
        # CNN for image processing
        self.image_network = nn.Sequential(
            # Input: (batch, 3, H, W) - 3 channels for object, color, state
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, self.image_shape[0], self.image_shape[1])
            cnn_output_size = self.image_network(dummy_input).shape[1]
        
        # Direction embedding
        self.direction_embedding = nn.Embedding(self.direction_dim, 8)
        
        # Combined feature size
        combined_feature_size = cnn_output_size + 8  # CNN features + direction embedding
        
        # RND for intrinsic motivation
        self.rnd = RNDNetwork(combined_feature_size)
        
        # LSTM for memory
        self.lstm = nn.LSTM(combined_feature_size, self.lstm_hidden_size)
        self.actor = nn.Linear(self.lstm_hidden_size, action_space.n)
        self.critic = nn.Linear(self.lstm_hidden_size, 1)

    def _process_observation(self, obs_dict):
        """Process Dict observation into features."""
        # Extract image and direction
        if isinstance(obs_dict, dict):
            image = obs_dict['image']
            direction = obs_dict['direction']
        else:
            # Fallback for legacy Box observations
            image = obs_dict
            direction = torch.zeros(obs_dict.shape[0], dtype=torch.long, device=obs_dict.device)
        
        # Process image: convert to float and permute to channels-first
        image_input = image.float().permute(0, 3, 1, 2)
        image_features = self.image_network(image_input)
        
        # Process direction
        direction_features = self.direction_embedding(direction.long())
        
        # Combine features
        combined_features = torch.cat([image_features, direction_features], dim=1)
        return combined_features

    def get_action_and_value(self, x, lstm_state, action=None):
        # Extract sequence length and batch size from the observation
        if isinstance(x, dict):
            seq_len, batch_size = x['image'].shape[0], x['image'].shape[1]
        else:
            seq_len, batch_size = x.shape[0], x.shape[1]
        
        # Handle Dict vs Box observations
        if isinstance(x, dict):
            # x is a dict observation
            x_reshaped = {
                'image': x['image'].view(seq_len * batch_size, *x['image'].shape[2:]),
                'direction': x['direction'].view(seq_len * batch_size)
            }
        else:
            # Legacy: x is tensor observations, convert to dict format
            x_reshaped = x.view(seq_len * batch_size, *self.image_shape)
            x_reshaped = {
                'image': x_reshaped,
                'direction': torch.zeros(seq_len * batch_size, dtype=torch.long, device=x.device)
            }
        
        # Process observations
        features = self._process_observation(x_reshaped)
        
        # Process through LSTM
        lstm_in = features.view(seq_len, batch_size, -1)
        lstm_out, lstm_state = self.lstm(lstm_in, lstm_state)
        
        logits = self.actor(lstm_out)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(lstm_out), lstm_state
    
    def get_intrinsic_reward(self, x):
        """Get intrinsic reward from RND for exploration."""
        # Extract sequence length and batch size from the observation
        if isinstance(x, dict):
            seq_len, batch_size = x['image'].shape[0], x['image'].shape[1]
        else:
            seq_len, batch_size = x.shape[0], x.shape[1]
        
        # Handle Dict vs Box observations
        if isinstance(x, dict):
            x_reshaped = {
                'image': x['image'].view(seq_len * batch_size, *x['image'].shape[2:]),
                'direction': x['direction'].view(seq_len * batch_size)
            }
        else:
            x_reshaped = x.view(seq_len * batch_size, *self.image_shape)
            x_reshaped = {
                'image': x_reshaped,
                'direction': torch.zeros(seq_len * batch_size, dtype=torch.long, device=x.device)
            }
        
        # Process observations and get intrinsic reward
        features = self._process_observation(x_reshaped)
        intrinsic_reward = self.rnd.intrinsic_reward(features)
        return intrinsic_reward.view(seq_len, batch_size)
    
    def update_rnd(self, x):
        """Update RND predictor network."""
        # Extract sequence length and batch size from the observation
        if isinstance(x, dict):
            seq_len, batch_size = x['image'].shape[0], x['image'].shape[1]
        else:
            seq_len, batch_size = x.shape[0], x.shape[1]
        
        # Handle Dict vs Box observations
        if isinstance(x, dict):
            x_reshaped = {
                'image': x['image'].view(seq_len * batch_size, *x['image'].shape[2:]),
                'direction': x['direction'].view(seq_len * batch_size)
            }
        else:
            x_reshaped = x.view(seq_len * batch_size, *self.image_shape)
            x_reshaped = {
                'image': x_reshaped,
                'direction': torch.zeros(seq_len * batch_size, dtype=torch.long, device=x.device)
            }
        
        # Process observations and get RND loss
        features = self._process_observation(x_reshaped)
        target_features, predicted_features = self.rnd(features)
        rnd_loss = torch.mean((target_features - predicted_features) ** 2)
        return rnd_loss