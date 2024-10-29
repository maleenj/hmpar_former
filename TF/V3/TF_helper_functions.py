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

def reverse_normalization(normalized_data, medians_per_joint_axis, iqrs_per_joint_axis):
    original_data = np.empty_like(normalized_data)  # Initialize an array to hold the original data

    # Iterate over each joint and each axis
    for joint in range(normalized_data.shape[0]):
        for axis in range(normalized_data.shape[1]):
            # Retrieve the median and IQR for this joint and axis
            median = medians_per_joint_axis[joint, axis]
            iqr = iqrs_per_joint_axis[joint, axis]

            # Retrieve the normalized values for this joint and axis
            normalized_values = normalized_data[joint, axis]

            # Calculate the original values based on the normalization formula
            original_values = (normalized_values * iqr) + median

            # Store the original values in the output array
            original_data[joint, axis] = original_values

    return original_data

def robust_normalize_data_with_clipping(data, medians_per_joint_axis, iqrs_per_joint_axis, normalized_data, clipping_percentiles=(1, 99)):
    for joint in range(data.shape[1]):  # For each joint
        for axis in range(data.shape[2]):  # For each axis (x, y, z)
            joint_axis_data = data[:, joint, axis]
            # Determine clipping thresholds based on percentiles
            lower_threshold, upper_threshold = np.percentile(joint_axis_data, clipping_percentiles)
            # Clip the data based on thresholds
            clipped_values = np.clip(joint_axis_data, lower_threshold, upper_threshold)
            # Normalize the clipped data, avoiding division by zero
            if iqrs_per_joint_axis[joint, axis] > 0:
                normalized_values = (clipped_values - medians_per_joint_axis[joint, axis]) / iqrs_per_joint_axis[joint, axis]
            else:
                normalized_values = clipped_values  # Keep original values if IQR is 0
            # Store the normalized values
            normalized_data[:, joint, axis] = normalized_values
    return normalized_data

def calculate_combined_statistics(data_list):
    combined_data = np.concatenate(data_list, axis=0)
    medians = np.median(combined_data, axis=0)
    q75, q25 = np.percentile(combined_data, [75, 25], axis=0)
    iqrs = q75 - q25
    return medians, iqrs

def process_datasets_with_combined_normalization(datasets, timestamps_list):
    results = {}
    pos_list, vel_list, acc_list = [], [], []

    # First pass: calculate velocity and acceleration for each dataset
    for i, (dataset, timestamps) in enumerate(zip(datasets, timestamps_list), 1):
        pos, vel, acc = calculate_velocity_acceleration(dataset, timestamps)
        pos_list.append(pos)
        vel_list.append(vel)
        acc_list.append(acc)

    # Calculate combined statistics
    medians_pos, iqrs_pos = calculate_combined_statistics(pos_list)
    medians_vel, iqrs_vel = calculate_combined_statistics(vel_list)
    medians_acc, iqrs_acc = calculate_combined_statistics(acc_list)

    # Second pass: normalize each dataset using the combined statistics
    for i, (pos, vel, acc) in enumerate(zip(pos_list, vel_list, acc_list), 1):
        norm_pos = np.empty_like(pos)
        norm_vel = np.empty_like(vel)
        norm_acc = np.empty_like(acc)

        norm_pos = robust_normalize_data_with_clipping(pos, medians_pos, iqrs_pos, norm_pos)
        norm_vel = robust_normalize_data_with_clipping(vel, medians_vel, iqrs_vel, norm_vel)
        norm_acc = robust_normalize_data_with_clipping(acc, medians_acc, iqrs_acc, norm_acc)

        results[f"dataset{i}_normpos"] = norm_pos
        results[f"dataset{i}_normvel"] = norm_vel
        results[f"dataset{i}_normacc"] = norm_acc

        print(f"Calculated and normalized for dataset{i}:")
        print(f"  Position shape: {norm_pos.shape}")
        print(f"  Velocity shape: {norm_vel.shape}")
        print(f"  Acceleration shape: {norm_acc.shape}")
        print()

    # Store the combined statistics
    results["combined_medians_pos"] = medians_pos
    results["combined_iqrs_pos"] = iqrs_pos
    results["combined_medians_vel"] = medians_vel
    results["combined_iqrs_vel"] = iqrs_vel
    results["combined_medians_acc"] = medians_acc
    results["combined_iqrs_acc"] = iqrs_acc

    return results
