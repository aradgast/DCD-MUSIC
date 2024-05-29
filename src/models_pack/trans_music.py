import torch
import torch.nn as nn
import math

# Internal imports
from src.system_model import SystemModel
from src.utils import *
from src.methods_pack.music import MUSIC


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(device) + self.pe[:x.size(0)].to(device)

        return x


class TransMUSIC(nn.Module):
    """

    """

    def __init__(self, system_model: SystemModel = None, mode: str = "batch_norm"):
        super(TransMUSIC, self).__init__()
        self.system_model = system_model
        self.params = self.system_model.params
        self.N = self.params.N
        self.estimation_params = None
        if self.params.field_type == "Far":
            self.estimation_params = "angle"
        elif self.params.field_type == "Near":
            self.estimation_params = "angle, range"
        self.music = MUSIC(system_model=self.system_model, estimation_parameter=self.estimation_params)

        if mode == "batch_norm":
            self.norm = nn.BatchNorm1d(self.N * 2).to(device)
        elif mode == "layer_norm":
            self.norm = nn.LayerNorm(normalized_shape=[self.N * 2]).to(device)
        else:
            raise ValueError(f"TransMUSIC.__init__: unrecognized mode {mode}")

        self.pos_encoder = PositionalEncoding(self.N * 2)  # Position embedding

        # Define encoder layer
        e_layer = nn.TransformerEncoderLayer(d_model=self.N * 2,
                                             nhead=self.N,  # The number of heads in a multi head attention model
                                             dim_feedforward=1024,  # Dimensions of feedforward network models
                                             dropout=0,  # Dropout value
                                             activation="relu",
                                             layer_norm_eps=1e-05,
                                             batch_first=True,
                                             norm_first=False,
                                             device=None,
                                             dtype=None).to(device)

        self.encoder = nn.TransformerEncoder(e_layer,
                                             num_layers=3,
                                             norm=None).to(device)

        self.input_linear = nn.Linear(in_features=self.N * 2, out_features=2 * self.N ** 2).to(device)

        self.output = nn.Sequential(

            nn.Linear(in_features=360, out_features=self.N * 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.N * 2, out_features=self.N * 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.N * 2, out_features=self.N * 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.N * 2, out_features=6)
        ).to(device)

        self.source_number_estimator = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=32, out_features=4)
        ).to(device)

    def forward(self, x, mode: str = "subspace_train"):
        N = self.N

        # The input X is [size, 16200] batch_ Size=16, input dimension N*2, sequence length T
        size = x.shape[0]  # Get batch size
        if mode == "subspace_train":
            x3 = self._get_noise_subspace(x)

            x4 = x3.reshape(size, N * 2, N).to(device)  # Change its mapping covariance to [size, N * 2, N]
            Un = torch.complex(x4[:, :N, :].to(device), x4[:, N:, :].to(device)).to(torch.complex32) # feature vector  [size, N, N]
            spectrum = self.music.get_music_spectrum_from_noise_subspace(Un).to(device)  # Calculate spectrum
            x7 = spectrum.float().to(device)
            predictions = self.output(x7.to(device)).to(device)
            with torch.no_grad():
                x9 = x3.detach()
                num_sources_est = self.source_number_estimator(x9)
            return predictions, num_sources_est

        elif mode == "num_source_train":
            with torch.no_grad():
                x3 = self._get_noise_subspace(x)
            x9 = x3.detach()
            num_sources_est = self.source_number_estimator(x9)

            return num_sources_est

        else:
            raise ValueError(f"TransMUSIC.forward: Unrecognized {mode}")

    def _get_noise_subspace(self, x):
        x = self.norm(x.to(torch.float32)).to(device)  # Become [size, 16200]

        x = x.permute(2, 0, 1).float().to(device)  # Exchange dimension becomes [T, size, N * 2]

        # Position embedding
        x = self.pos_encoder(x.to(device)).to(device)  # x: Tensor, shape [seq_len, batch_size, embedding_dim]
        x = x.permute(1, 0, 2).float().to(device)  # Exchange dimension becomes [size, T, N * 2]

        x1 = self.encoder(x.to(device))  # Transformer_ Encoder network output becomes [size, T, N * 2]
        x2 = torch.mean(x1, dim=1)  # Output becomes [size, N * 2]
        x3 = self.input_linear(x2).to(device)
        return x3

    def pre_processing(self, x: torch.Tensor):
        """
        The input data is a complex signal of size [batch_size, N, T] and the input of the model need to be real signal
        of size [batch_size, 2N, T]
        """
        x = torch.concatenate([x.real, x.imag], dim=1)
        return x
