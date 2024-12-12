import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCNN(nn.Module):
    def __init__(self, num_classes=10, num_lstm_hidden=128, num_lstm_layers=1):
        super(LSTMCNN, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size=256 * 4 * 4, hidden_size=num_lstm_hidden, 
                            num_layers=num_lstm_layers, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(num_lstm_hidden, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Input x shape: [batch_size, time_steps, channels, height, width]
        x = x.unsqueeze(1)  # Add channel dimension
        batch_size, time_steps, C, H, W = x.size()
        
        # Apply CNN layers on each time step
        cnn_features = []
        for t in range(time_steps):
            out = self.pool(F.relu(self.conv1(x[:, t, :, :, :])))
            out = self.pool(F.relu(self.conv2(out)))
            out = self.pool(F.relu(self.conv3(out)))
            out = out.view(out.size(0), -1)  # Flatten: [batch_size, 256 * 4 * 4]
            cnn_features.append(out)
        
        # Stack CNN features along time axis
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch_size, time_steps, 256 * 4 * 4]
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)  # [batch_size, time_steps, num_lstm_hidden]
        lstm_out = lstm_out[:, -1, :]  # Take the output from the last time step
        
        # Fully connected layers
        x = F.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        
        return x
