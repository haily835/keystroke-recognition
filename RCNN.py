import torch
import torch.nn as nn
from torchsummary import summary


class RCNN(nn.Module):
    def __init__(self, feature_channel_number, hidden_size, output_size):
        super(RCNN, self).__init__()
        
        # First 3D convolutional layer: https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        # Input shape: (Batch size, in_channels, Depth, Height, Witdth)
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
       
        x = x.view(-1, 2, x.size(1), 21, 3)
        
        # First 3D convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        print("After 1st CNN layer:", x.size())
        # Second 3D convolutional layer
        x = self.conv2(x)
        print("After 2nd CNN layer:", x.size())
        # Reshape for GRU

        x = x.view(x.size(0), x.size(1), -1)
        print(x.size())
        # First GRU layer
        h1_prev = torch.zeros(1, x.size(0), self.gru1.hidden_size).to(x.device)
        outputs = []
        for j in range(x.size(1)):
            input1 = torch.cat((x[:, j], h1_prev.squeeze(0)), dim=1).unsqueeze(1)
            _, h1 = self.gru1(input1, h1_prev)
            h1_prev = h1
            outputs.append(h1)
        outputs = torch.cat(outputs, dim=1)
 
        # # Second GRU layer
        # h2_prev = torch.zeros(1, x.size(0), self.gru2.hidden_size).to(x.device)
        # outputs, _ = self.gru2(outputs, h2_prev)
        
        # # Fully connected layers
        # outputs = self.fc2(self.relu(self.fc1(outputs)))
        
        return x

# Example usage
# feature_channel_number = 256  # Feature channel number from the second convolutional layer
# hidden_size = 256  # Hidden size for GRU layers
# output_size = 28  # Output dimension
# model = RCNN(feature_channel_number, hidden_size, output_size)
# # Reshape the input data to batch x 2 x window size x 21 x 3
# # Print the model architecture
batch_size = 1
window_size = 30
# summary(model, (window_size, 2 , 21, 3), batch_size)


input = torch.randn(batch_size, 2, window_size, 21, 3)

# Input shape: (Batch size, in_channels, Depth, Height, Witdth)
conv3D_1 = nn.Conv3d(in_channels=2, out_channels=256, kernel_size=(3, 4, 3), padding=(1, 3, 0), stride=(1, 4, 1))
output = conv3D_1(input)
print("1st CNN:", output.shape)
conv3D_2 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 6, 1), padding=0, stride=1)
output = conv3D_2(output)
print("2nd CNN:", output.shape)
# torch.Size([1, 256, 5, 6, 1]) . Expected N x 256 x 6 x 1 => 5 x 256 x 6 x 1