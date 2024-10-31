import torch.nn as nn
import torch
from src.system_model import SystemModel

EIGEN_REGULARIZATION_WEIGHT = 100

class ParentModel(nn.Module):
    def __init__(self, system_model: SystemModel):
        super(ParentModel, self).__init__()
        self.system_model = system_model

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

    def forward(self, batch, batch_idx):
        raise NotImplementedError

    def set_eigenregularization_schedular(self, init_value=EIGEN_REGULARIZATION_WEIGHT, step_size=10, gamma=0.5):
        self.schedular_counter = 0
        self.schedular_step_size = step_size
        self.schedular_gamma = gamma
        self.eigenregularization_weight = init_value
        self.schedular_acc_current = 0
        self.schedular_patience = 10
        self.schedular_patience_counter = 0

    def update_eigenregularization_weight(self, acc):
        if acc <= self.schedular_acc_current:
            self.schedular_acc_current = acc
            self.schedular_patience_counter = 0
        else:
            self.schedular_patience_counter += 1
            if self.schedular_patience_counter >= self.schedular_patience:
                self.eigenregularization_weight *= self.schedular_gamma
                self.schedular_patience_counter = 0
                print(f"\nEigenregularization weight updated to {self.eigenregularization_weight}")

        # if self.schedular_counter % self.schedular_step_size == 0 and self.schedular_counter != 0:
        #     self.eigenregularization_weight *= self.schedular_gamma
        #     print(f"\nEigenregularization weight updated to {self.eigenregularization_weight}")
        # self.schedular_counter += 1

    def source_estimation_accuracy(self, sources_num, source_estimation):
        return torch.sum(source_estimation == sources_num * torch.ones_like(source_estimation).float()).item()


if __name__ == "__main__":
    pass