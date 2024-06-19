"""
Inspired by the paper:
J. Ji, W. Mao, F. Xi and S. Chen,
"TransMUSIC: A Transformer-Aided Subspace Method for DOA Estimation with Low-Resolution ADCS,"
 ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea,
 Republic of, 2024, pp. 8576-8580, doi: 10.1109/ICASSP48485.2024.10447483. keywords:
  {Direction-of-arrival estimation;Quantization (signal);Acoustic distortion;Simulation;Estimation;Signal processing algorithms
  ;Transformers;DOA estimation;Transformer;low-resolution ADCs;quantization},


"""

import torch
import torch.nn as nn
import math

# Internal imports
from src.system_model import SystemModel
from src.utils import *
from src.methods_pack.music import MUSIC


class ShiftedReLU(nn.ReLU):
    def __init__(self, shift=0.5):
        super(ShiftedReLU, self).__init__()
        self.shift = shift

    def forward(self, input):
        return super(ShiftedReLU, self).forward(input) + self.shift


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
        if self.estimation_params == "angle":
            input_dim = self.music.angels.shape[0]
            if self.music.system_model.params.M is not None:
                output_dim = self.music.system_model.params.M
            else:
                output_dim = self.music.system_model.params.N - 1
        elif self.estimation_params == "angle, range":
            input_dim = self.music.angels.shape[0] * self.music.distances.shape[0]
            if self.music.system_model.params.M is not None:
                output_dim = self.music.system_model.params.M * 2
            else:
                output_dim = (self.music.system_model.params.N - 1) * 2
            # self.activation = ShiftedReLU(shift=np.floor(self.music.system_model.fresnel)).to(device)
            # self.activation = nn.ReLU().to(device)
            self.activation = nn.LeakyReLU(negative_slope=0.0001).to(device)
        else:
            raise ValueError(f"TransMUSIC.__init__: unrecognized estimation parameter {self.estimation_params}")
        self.output = nn.Sequential(

            nn.Linear(in_features=input_dim, out_features=self.N * 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.N * 2, out_features=self.N * 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.N * 2, out_features=self.N * 2),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=self.N * 2, out_features=output_dim)
        ).to(device)

        self.source_number_estimator = nn.Sequential(
            nn.Linear(in_features=2 * self.N ** 2, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=32, out_features=self.params.N - 1),
            nn.Softmax(dim=1)
        ).to(device)

    def forward(self, x, mode: str = "subspace_train"):
        N = self.N
        x = self.pre_processing(x)
        # The input X is [size, 16200] batch_ Size=16, input dimension N*2, sequence length T

        size = x.shape[0]  # Get batch size

        x3 = self._get_noise_subspace(x)
        if mode == "subspace_train":
            x4 = x3.reshape(size, N * 2, N).to(device)  # Change its mapping covariance to [size, N * 2, N]
            Un = torch.complex(x4[:, :N, :], x4[:, N:, :]).to(torch.complex128)  # feature vector  [size, N, N]
            spectrum = self.music.get_music_spectrum_from_noise_subspace(Un)  # Calculate spectrum
            x7 = spectrum.float().to(device)
            x7 = x7.view(size, -1).to(device)  # Change the shape of the spectrum to [size, N * 2]
            predictions = self.output(x7).to(device)
            if self.estimation_params == "angle, range":
                angles, distances = torch.split(predictions, predictions.shape[-1] // 2, dim=1)
                distances = self.activation(distances).to(device)
                predictions = torch.cat([angles, distances], dim=1).to(device)
            with torch.no_grad():
                x9 = x3.detach()
                prob_sources_est = self.source_number_estimator(x9)
        elif mode == "num_source_train":
            prob_sources_est = self.source_number_estimator(x3)
            with torch.no_grad():
                x4 = x3.detach().reshape(size, N * 2, N).to(device)  # Change its mapping covariance to [size, N * 2, N]
                Un = torch.complex(x4[:, :N, :], x4[:, N:, :]).to(torch.complex32)  # feature vector  [size, N, N]
                spectrum = self.music.get_music_spectrum_from_noise_subspace(Un)  # Calculate spectrum
                x7 = spectrum.float().to(device)
                x7 = x7.view(size, -1).to(device)  # Change the shape of the spectrum to [size, N * 2]
                predictions = self.output(x7).to(device)
                if self.estimation_params == "angle, range":
                    angles, distances = torch.split(predictions, predictions.shape[-1] // 2, dim=1)
                    distances = self.activation(distances).to(device)
                    predictions = torch.cat([angles, distances], dim=1).to(device)
        else:
            raise ValueError(f"TransMUSIC.forward: Unrecognized {mode}")
        return predictions, prob_sources_est

    def _get_noise_subspace(self, x):
        x = self.norm(x.to(torch.float32)).to(device)  # Become [size, N * 2, T]

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
        if x.dim() == 2:
            x = x[None, :, :]
        x = torch.cat([x.real, x.imag], dim=1)
        return x

    def get_model_file_name(self):
        if self.system_model.params.M is None:
            M = "rand"
        else:
            M = self.system_model.params.M
        return f"TransMUSIC_" + \
            f"N={self.N}_" + \
            f"M={M}_" + \
            f"{self.system_model.params.signal_type}_" + \
            f"SNR={self.system_model.params.snr}_" + \
            f"{self.system_model.params.field_type}_field_" + \
            f"{self.system_model.params.signal_nature}"

    def get_model_name(self):
        return "TransMUSIC"

    def get_model_params(self):
        return None
