import warnings

import torch.nn as nn
import torch
from src.system_model import SystemModel

EIGEN_REGULARIZATION_WEIGHT = 50
class ParentModel(nn.Module):
    def __init__(self, system_model: SystemModel):
        super(ParentModel, self).__init__()
        self.system_model = system_model
        self.under_estimation_counter = 0
        self.over_estimation_counter = 0

    def get_model_name(self):
        return f"{self._get_name()}_{self.print_model_params()}"

    def print_model_params(self):
        return None

    def get_model_params(self):
        return None

    def get_model_file_name(self):
        if self.system_model.params.M is None:
            M = "rand"
        else:
            M = self.system_model.params.M
        return f"{self.get_model_name()}_" + \
            f"N={self.N}_" + \
            f"M={M}_" + \
            f"T={self.system_model.params.T}_" + \
            f"{self.system_model.params.signal_type}_" + \
            f"SNR={self.system_model.params.snr}_" + \
            f"{self.system_model.params.field_type}_field_" + \
            f"{self.system_model.params.signal_nature}_" + \
            f"eta={self.system_model.params.eta}_" + \
            f"sv_var={self.system_model.params.sv_noise_var}"

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def set_eigenregularization_schedular(self, init_value=EIGEN_REGULARIZATION_WEIGHT, step_size=10, gamma=0.5):
        self.schedular_counter = 0
        self.schedular_step_size = step_size
        self.schedular_gamma = gamma
        self.eigenregularization_weight = init_value
        if self.field_type == "Far":
            self.eigenregularization_weight /= 50
            self.schedular_gamma = 0.1

        self.schedular_acc_current = 0
        self.schedular_patience_ascending = 10
        self.schedular_patience_descending = 10
        self.schedular_patience_counter_descending = 0
        self.schedular_patience_counter_ascending = 0

        self.schedular_low_threshold = 70
        self.schedular_high_threshold = 90
        self.schedular_min_weight = 0.01
        self.schedular_max_weight = init_value * 2

    def get_eigenregularization_weight(self):
        return self.eigenregularization_weight

    def update_eigenregularization_weight(self, acc):

        # rounded_acc = round(acc, 1)
        # rounded_acc = round(acc, 0)
        rounded_acc = acc // 5 * 5

        if rounded_acc > self.schedular_acc_current or rounded_acc >= self.schedular_high_threshold:
            self.schedular_patience_counter_ascending += 1
            self.schedular_patience_counter_descending = 0
            if self.schedular_patience_counter_ascending >= self.schedular_patience_ascending:
                self.eigenregularization_weight = max(self.schedular_min_weight,
                                                      self.eigenregularization_weight * self.schedular_gamma)
                self.schedular_patience_counter_ascending = 0

        elif rounded_acc <= self.schedular_acc_current or rounded_acc <= self.schedular_low_threshold:
            self.schedular_patience_counter_descending += 1
            if acc > 95:
                self.schedular_patience_counter_descending += 1
            self.schedular_patience_counter_ascending = 0
            if self.schedular_patience_counter_descending >= self.schedular_patience_descending:
                self.eigenregularization_weight = min(self.schedular_max_weight,
                                                      self.eigenregularization_weight / self.schedular_gamma)
                self.schedular_patience_counter_descending = 0
        else:
            self.schedular_patience_counter_ascending = 0
            self.schedular_patience_counter_descending = 0

        self.schedular_acc_current = rounded_acc


        # if self.schedular_counter % self.schedular_step_size == 0 and self.schedular_counter != 0:
        #     self.eigenregularization_weight *= self.schedular_gamma
        #     print(f"\nEigenregularization weight updated to {self.eigenregularization_weight}")
        # self.schedular_counter += 1

    def source_estimation_accuracy(self, sources_num, source_estimation):
        if not self.training:
            if (sources_num < source_estimation).any():
                self.over_estimation_counter += sum(sources_num < source_estimation).item()
            if (sources_num > source_estimation).any():
                self.under_estimation_counter += sum(sources_num > source_estimation).item()
        return torch.sum(source_estimation == sources_num * torch.ones_like(source_estimation).float()).item()


if __name__ == "__main__":
    pass