import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import sys
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split


def generate_sequences(norm_pos, norm_vel, norm_acc, input_length=60, predict_length=60):
    num_frames = norm_pos.shape[0]
    num_joints = norm_pos.shape[1]

    # Calculate the total number of sequences we can create
    num_sequences = num_frames - input_length - predict_length + 1

    # Initialize arrays to store the input and target sequences
    X_pos = np.zeros((num_sequences, input_length, num_joints, 3))
    X_vel = np.zeros((num_sequences, input_length, num_joints, 3))
    X_acc = np.zeros((num_sequences, input_length, num_joints, 3))
    Y_pos = np.zeros((num_sequences, predict_length, num_joints, 3))
    Y_vel = np.zeros((num_sequences, predict_length, num_joints, 3))
    Y_acc = np.zeros((num_sequences, predict_length, num_joints, 3))

    # Create sequences
    for i in range(num_sequences):
        X_pos[i] = norm_pos[i:i + input_length]
        X_vel[i] = norm_vel[i:i + input_length]
        X_acc[i] = norm_acc[i:i + input_length]
        Y_pos[i] = norm_pos[i + input_length:i + input_length + predict_length]
        Y_vel[i] = norm_vel[i + input_length:i + input_length + predict_length]
        Y_acc[i] = norm_acc[i + input_length:i + input_length + predict_length]

    return X_pos, X_vel, X_acc, Y_pos, Y_vel, Y_acc


def create_shifted_mask(seq_length, num_joints):
    # seq_length is the number of time steps
    # num_joints is the number of joints per time step
    total_length = seq_length * num_joints
    mask = torch.ones((total_length, total_length), dtype=torch.float32) * float('-inf')  # Start with everything masked
    for i in range(seq_length):
        for j in range(i + 1):  # Allow visibility up to and including the current time step
            start_row = i * num_joints
            end_row = start_row + num_joints
            start_col = j * num_joints
            end_col = start_col + num_joints
            mask[start_row:end_row, start_col:end_col] = 0.0  # Unmask the allowed region

    return mask

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, output, target):
        # Compute the squared differences
        squared_diff = (output - target) ** 2

        # Calculate the mean of the squared differences
        loss = squared_diff.mean()

        return loss
