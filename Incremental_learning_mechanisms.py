import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# MECHANISM 1: Continual backpropagation
# Logic: Prevents "plasticity loss" in non-stationary streams by reinitializing
# low-utility neurons based on age and contribution.
# =============================================================================

class ContinualBackpropModule:
    def __init__(self, model, width, age_threshold=20, reinit_rate=0.05, alpha=0.6):
        """
        Args:
            model: The KAN or Neural Network model.
            width: List containing the number of units in each hidden layer.
            age_threshold: Minimum steps before a neuron is eligible for replacement.
            reinit_rate: Fraction of eligible low-utility neurons to reinitialize.
            alpha: Smoothing factor for utility and activation tracking.
        """
        self.model = model
        self.width = width
        self.age_threshold = age_threshold
        self.reinit_rate = reinit_rate
        self.alpha = alpha

        # Tracking buffers for each hidden layer
        self.unit_ages = [torch.zeros(n) for n in width]
        self.unit_utilities = [torch.zeros(n) for n in width]
        self.mean_activations = [torch.zeros(n) for n in width]

    def update_and_reinit(self, activations):
        """
        Performs neuron utility evaluation and selective reinitialization.
        Args:
            activations: List of activation tensors from the model's forward pass.
        """
        for l in range(len(self.width)):
            self.unit_ages[l] += 1

            # 1. Track Mean Activations
            curr_acts = activations[l].detach().mean(dim=0)
            self.mean_activations[l] = (self.alpha * self.mean_activations[l] +
                                        (1 - self.alpha) * curr_acts)

            # 2. Calculate Neuron Utility
            # Utility = (Contribution to Output) * (Adaptability of Input)
            layer = self.model.model.act_fun[l]
            in_weights = layer.input_weights.weight.data
            out_weights = layer.output_weights.weight.data

            diff_from_mean = torch.abs(curr_acts - self.mean_activations[l])
            contribution = diff_from_mean * torch.sum(torch.abs(out_weights), dim=0)
            adaptation = 1.0 / (torch.sum(torch.abs(in_weights), dim=1) + 1e-6)

            utility = contribution * adaptation
            self.unit_utilities[l] = (self.alpha * self.unit_utilities[l] +
                                      (1 - self.alpha) * utility)

            # 3. Selective Reinitialization
            eligible_mask = self.unit_ages[l] > self.age_threshold
            eligible_indices = torch.where(eligible_mask)[0]

            if len(eligible_indices) > 0:
                num_reinit = int(self.reinit_rate * len(eligible_indices))
                if num_reinit > 0:
                    _, low_util_idx = torch.topk(self.unit_utilities[l][eligible_indices],
                                                 k=num_reinit, largest=False)
                    reinit_indices = eligible_indices[low_util_idx]

                    self._reset_neurons(l, reinit_indices, in_weights, out_weights)

    def _reset_neurons(self, layer_idx, indices, in_weights, out_weights):
        """Resets weights and metadata for specific neurons."""
        with torch.no_grad():
            # Reinitialize input weights with noise
            in_weights[indices] = torch.randn_like(in_weights[indices])
            # Zero out output weights to prevent sudden loss spikes
            out_weights[:, indices] = 0

            # Reset metadata
            self.unit_ages[layer_idx][indices] = 0
            self.unit_utilities[layer_idx][indices] = 0
            self.mean_activations[layer_idx][indices] = 0

# =============================================================================
# MECHANISM 2 and 3: Dynamic sliding window mechanism and Dynamic experience replay mechanism
# Logic: Dynamically adjusts the training window size based on the rate of
# change in validation loss to handle hydrological non-stationarity.
# =============================================================================

class DynamicWindowAdjuster:
    def __init__(self, min_w=5, max_w=25, lambda_adj=0.4, alpha_smooth=0.6):
        """
        Args:
            min_w: Minimum allowable window size.
            max_w: Maximum allowable window size.
            lambda_adj: Sensitivity factor for loss changes.
            alpha_smooth: Smoothing factor for the adjusted window size.
        """
        self.min_w = min_w
        self.max_w = max_w
        self.lambda_adj = lambda_adj
        self.alpha_smooth = alpha_smooth
        self.last_val_loss = None

    def adjust(self, current_val_loss, current_w, smoothed_w):
        """
        Args:
            current_val_loss: Focused MSE from the latest validation step.
            current_w: The current raw window size.
            smoothed_w: The current smoothed window size.
        Returns:
            new_w: The updated raw window size.
            new_smoothed_w: The updated smoothed window size.
        """
        if self.last_val_loss is None:
            self.last_val_loss = current_val_loss
            return current_w, smoothed_w

        # 1. Calculate relative loss change (delta_L)
        delta_L = (current_val_loss - self.last_val_loss) / (self.last_val_loss + 1e-6)

        # 2. Adjust window size: Inverse relationship with loss change
        # If loss increases (error up), shrink window to focus on recent data.
        # If loss decreases (learning well), expand window for stability.
        new_w = np.clip(current_w * (1 - self.lambda_adj * delta_L),
                        self.min_w, self.max_w)

        # 3. Apply smoothing to prevent oscillation
        new_smoothed_w = (self.alpha_smooth * smoothed_w +
                          (1 - self.alpha_smooth) * new_w)

        self.last_val_loss = current_val_loss
        return new_w, new_smoothed_w