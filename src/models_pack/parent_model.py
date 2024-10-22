import torch.nn as nn
from src.system_model import SystemModel


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

    def loss(self, *args, **kwargs):
        raise NotImplementedError

if __name__ == "__main__":
    pass