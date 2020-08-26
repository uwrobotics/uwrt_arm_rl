import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DQN(nn.Module):
    def __init__(self, in_channel, hidden_channels,  out_channel, kernel_sizes, strides,
                 observation_space_h, observation_space_w, action_space):
        super(DQN, self).__init__()
        # 1st block
        self.conv1 = nn.Conv1d(in_channel, hidden_channels[0], kernel_size=kernel_sizes[0], stride=strides[0])
        self.bn1 = nn.BatchNorm1d(hidden_channels[0])
        # 2nd block
        self.conv2 = nn.Conv1d(hidden_channels[0], hidden_channels[1], kernel_size=kernel_sizes[1], stride=strides[1])
        self.bn2 = nn.BatchNorm1d(hidden_channels[1])
        # 3rd block
        self.conv3 = nn.Conv1d(hidden_channels[1], hidden_channels[1], kernel_size=kernel_sizes[2], stride=strides[2])

        ######################################################
        # Manually calc number of Linear input connections
        # which depends on output of conv1d layers
        ######################################################
        def conv1d_size_out(size, kernel_size=1, stride=1):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv1d_size_out(conv1d_size_out(conv1d_size_out(observation_space_w,kernel_sizes[0],strides[0]),
                                                kernel_sizes[1],strides[1]),
                                                kernel_sizes[2],strides[2])
        convh = conv1d_size_out(conv1d_size_out(conv1d_size_out(observation_space_h,kernel_sizes[0],strides[0]),
                                                kernel_sizes[1],strides[1]),
                                                kernel_sizes[2],strides[2])

        #######################
        linear_input_size = convw * convh * hidden_channels[1]
        self.linear = nn.Linear(linear_input_size, out_channel)
        self.head = nn.Linear(out_channel, action_space)

    # Called with either one element to determine next action, or a batch during optimization.
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear(x.view(x.size(0), -1)))
        return self.head(x)