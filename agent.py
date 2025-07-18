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
        
        self.image_shape = observation_space.shape  # (H, W, 3) where 3 = (OBJECT, COLOR, STATE)
        self.lstm_hidden_size = 128  # Restored to reasonable size
        
        # Improved CNN - not too simple, not too complex
        self.network = nn.Sequential(
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
            cnn_output_size = self.network(dummy_input).shape[1]
        
        # RND for intrinsic motivation
        self.rnd = RNDNetwork(cnn_output_size)
        
        # Lightweight LSTM for memory
        self.lstm = nn.LSTM(cnn_output_size, self.lstm_hidden_size)
        self.actor = nn.Linear(self.lstm_hidden_size, action_space.n)
        self.critic = nn.Linear(self.lstm_hidden_size, 1)

    def get_action_and_value(self, x, lstm_state, action=None):
        seq_len, batch_size = x.shape[0], x.shape[1]
        x_reshaped = x.view(seq_len * batch_size, *self.image_shape)
        
        # Convert to float and permute to channels-first: (batch, channels, H, W)
        x_input = x_reshaped.float().permute(0, 3, 1, 2)
        
        # Process through CNN
        features = self.network(x_input)
        
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
        seq_len, batch_size = x.shape[0], x.shape[1]
        x_reshaped = x.view(seq_len * batch_size, *self.image_shape)
        
        # Convert to float and permute to channels-first: (batch, channels, H, W)
        x_input = x_reshaped.float().permute(0, 3, 1, 2)
        
        # Process through CNN
        features = self.network(x_input)
        
        # Get intrinsic reward
        intrinsic_reward = self.rnd.intrinsic_reward(features)
        return intrinsic_reward.view(seq_len, batch_size)
    
    def update_rnd(self, x):
        """Update RND predictor network."""
        seq_len, batch_size = x.shape[0], x.shape[1]
        x_reshaped = x.view(seq_len * batch_size, *self.image_shape)
        
        # Convert to float and permute to channels-first: (batch, channels, H, W)
        x_input = x_reshaped.float().permute(0, 3, 1, 2)
        
        # Process through CNN
        features = self.network(x_input)
        
        # Get RND loss
        target_features, predicted_features = self.rnd(features)
        rnd_loss = torch.mean((target_features - predicted_features) ** 2)
        return rnd_loss