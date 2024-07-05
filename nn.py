import torch
torch.manual_seed(10)
import torch.nn.functional as F
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, feat_size, num_classes):
        '''
        Initialize the neural net with the linear layers. We require your model to have exactly two 
        linear layers to pass the autograder and that you adhere to the specified input and output shapes. 
        You are free to design the rest of your network as you see fit and use any non-linearity or 
        dropout layers to obtain the necessary accuracy.
    
        Args:
            feat_size: number of features of one input datapoint
            num_classes: Number of classes (labels)        
        '''
        super(NN, self).__init__()
        self.fc1 = nn.Linear(feat_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        '''
        Implement the forward function to feed the input x through the model and get the output.

        Args:
            x: (B, D) tensor containing the encoded data where B = batch size and D = feat_size

        Returns:
            output: (B, C) tensor of logits where B = batch size and C = num_classes

        '''
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)

        return x
