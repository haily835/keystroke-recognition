import torch
import torch.nn as nn

class RCNN(nn.Module):
    def __init__(self, feature_channel_number, hidden_size, output_size):
        super(RCNN, self).__init__()
        
        # First 3D convolutional layer
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=256, kernel_size=(3, 4, 3), padding=(1, 3, 0), stride=(1, 4, 1))
        self.relu = nn.ReLU()
        
        # Second 3D convolutional layer
        self.conv2 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 6, 1), padding=0, stride=1)
        
        # First GRU layer
        self.gru1 = nn.GRU(input_size=feature_channel_number, hidden_size=hidden_size, batch_first=True)
        
        # Second GRU layer
        self.gru2 = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Reshape the input data to batch x 2 x window size x 21 x 3
        x = x.view(-1, 2, x.size(1), 21, 3)
        
        # First 3D convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        
        # Second 3D convolutional layer
        x = self.conv2(x)
        
        # Reshape for GRU
        x = x.view(x.size(0), x.size(1), -1)
        
        # First GRU layer
        h1_prev = torch.zeros(1, x.size(0), self.gru1.hidden_size).to(x.device)
        outputs = []
        for j in range(x.size(1)):
            input1 = torch.cat((x[:, j], h1_prev.squeeze(0)), dim=1).unsqueeze(1)
            _, h1 = self.gru1(input1, h1_prev)
            h1_prev = h1
            outputs.append(h1)
        outputs = torch.cat(outputs, dim=1)
        
        # Second GRU layer
        h2_prev = torch.zeros(1, x.size(0), self.gru2.hidden_size).to(x.device)
        outputs, _ = self.gru2(outputs, h2_prev)
        
        # Fully connected layers
        outputs = self.fc2(self.relu(self.fc1(outputs)))
        
        return outputs

# Example usage
feature_channel_number = 256  # Feature channel number from the second convolutional layer
hidden_size = 64  # Hidden size for GRU layers
output_size = 28  # Output dimension
model = RCNN(feature_channel_number, hidden_size, output_size)

# Print the model architecture
print(model)
