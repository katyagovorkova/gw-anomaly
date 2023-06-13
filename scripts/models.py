import torch.nn as nn
import torch.nn.functional as F
from torch import transpose as torchtranspose
from math import floor
from functools import reduce
import torch

class DNN_AE(nn.Module):
    def __init__(self, num_ifos, num_timesteps, BOTTLENECK, FACTOR):
        super(DNN_AE, self).__init__()
        print("WARNING: Change this with Eric's actual LSTM model!")
        self.num_timesteps = num_timesteps
        self.num_ifos = num_ifos
        self.Conv1 = nn.Conv1d(in_channels = num_ifos, out_channels = 5, kernel_size=5, padding='same')
        self.Linear1 = nn.Linear(num_timesteps*5, 100)
        self.Linear2 = nn.Linear(100, BOTTLENECK)
        self.Linear3 = nn.Linear(BOTTLENECK, num_timesteps*5)
        self.Conv2 = nn.Conv1d(in_channels=5, out_channels = 2, kernel_size = 5, padding = 'same')
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.Conv1(x))
        x = x.view(-1, self.num_timesteps*5)
        x = F.tanh(self.Linear1(x))
        x = F.tanh(self.Linear2(x))
        x = F.tanh(self.Linear3(x))
        x = x.view(batch_size, 5, self.num_timesteps)
        x = self.Conv2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ, out_activ):
        super(Encoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [out_dim]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ, self.out_activ = h_activ, out_activ

    def forward(self, x):
        x = x.unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)
            elif self.out_activ and index == self.num_layers - 1:
                return self.out_activ(h_n).squeeze()

        return h_n.squeeze()


class Decoder(nn.Module):
    def __init__(self, input_dim, out_dim, h_dims, h_activ):
        super(Decoder, self).__init__()

        layer_dims = [input_dim] + h_dims + [h_dims[-1]]
        self.num_layers = len(layer_dims) - 1
        self.layers = nn.ModuleList()
        for index in range(self.num_layers):
            layer = nn.LSTM(
                input_size=layer_dims[index],
                hidden_size=layer_dims[index + 1],
                num_layers=1,
                batch_first=True,
            )
            self.layers.append(layer)

        self.h_activ = h_activ
        self.dense_matrix = nn.Parameter(
            torch.rand((layer_dims[-1], out_dim), dtype=torch.float), requires_grad=True
        )

    def forward(self, x, seq_len):
        x = x.repeat(seq_len, 1).unsqueeze(0)
        for index, layer in enumerate(self.layers):
            x, (h_n, c_n) = layer(x)

            if self.h_activ and index < self.num_layers - 1:
                x = self.h_activ(x)

        return torch.mm(x.squeeze(0), self.dense_matrix)


######
# MAIN
######


class LSTM_AE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoding_dim,
        h_dims=[],
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
    ):
        super(LSTM_AE, self).__init__()

        self.encoder = Encoder(input_dim, encoding_dim, h_dims, h_activ, out_activ)
        self.decoder = Decoder(encoding_dim, input_dim, h_dims[::-1], h_activ)

    def forward(self, x):
        seq_len = x.shape[0]
        x = self.encoder(x)
        x = self.decoder(x, seq_len)

        return x

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1).squeeze()


class UnFlatten(nn.Module):
    def __init__(self, in_channels, input_dims):
        super(UnFlatten, self).__init__()

        self.in_channels, self.input_dims = in_channels, input_dims

    def forward(self, x):
        return x.reshape((1, self.in_channels, *self.input_dims))


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dim):
        super(ConvUnit, self).__init__()

        # TODO: Handle dim == 1
        conv = nn.Conv3d if dim == 3 else nn.Conv2d
        self.conv = conv(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)

        return x


class DeConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dim):
        super(DeConvUnit, self).__init__()

        # TODO: Handle dim == 1
        deconv = nn.ConvTranspose3d if dim == 3 else nn.ConvTranspose2d
        self.deconv = deconv(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride
        )
        self.relu = nn.ReLU()

    def forward(self, x, output_size):
        x = self.deconv(x, output_size=output_size)
        # x = self.relu(x)

        return x


###########
# UTILITIES
###########


def compute_output_dim(num_layers, input_dim, kernel, stride, out_dims=[]):
    if not num_layers:
        return out_dims

    # Guide to convolutional arithmetic: https://arxiv.org/pdf/1603.07285.pdf
    out_dim = floor((input_dim - kernel) / stride) + 1
    out_dims.append(out_dim)

    return compute_output_dim(num_layers - 1, out_dim, kernel, stride, out_dims)


######
# MAIN
######


class CONV_AE(nn.Module):
    def __init__(self, input_dims, encoding_dim, kernel, stride, in_channels=1,
                 h_channels=[1]):
        super(CONV_AE, self).__init__()

        conv_dim = len(input_dims)
        all_channels = [in_channels] + h_channels
        num_layers = len(all_channels) - 1

        if isinstance(kernel, int):
            kernel = (kernel, ) * conv_dim
        if isinstance(stride, int):
            stride = (stride, ) * conv_dim

        out_dims = []
        for i, k, s in zip(input_dims, kernel, stride):
            out_dims.append(compute_output_dim(num_layers, i, k, s, []))
        out_dims = [input_dims] + list(zip(*out_dims))

        self.out_dims = out_dims[::-1]
        out_dims = self.out_dims[0]
        flat_dim = all_channels[-1] * reduce(lambda x, y: x * y, out_dims)

        # Construct encoder and decoder units
        encoder_layers = []
        self.decoder_layers = nn.ModuleList([
            nn.Linear(encoding_dim, flat_dim),
            UnFlatten(all_channels[-1], out_dims)
        ])
        for index in range(num_layers):
            conv_layer = ConvUnit(
                in_channels=all_channels[index],
                out_channels=all_channels[index + 1],
                kernel=kernel,
                stride=stride,
                dim=conv_dim
            )
            deconv_layer = DeConvUnit(
                in_channels=all_channels[-index - 1],
                out_channels=all_channels[-index - 2],
                kernel=kernel,
                stride=stride,
                dim=conv_dim
            )

            encoder_layers.append(conv_layer)
            self.decoder_layers.append(deconv_layer)

        encoder_layers.extend([Flatten(),
                               nn.Linear(flat_dim, encoding_dim)])
        self.encoder = nn.Sequential(*encoder_layers)

    def decoder(self, x):
        for index, layer in enumerate(self.decoder_layers):
            if isinstance(layer, DeConvUnit):
                x = layer(x, output_size=self.out_dims[1:][index - 2])
            else:
                x = layer(x)

        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    

class CONV_LSTM_AE(nn.Module):
    def __init__(self, input_dims, encoding_dim, kernel, stride=1,
                 h_conv_channels=[1], h_lstm_channels=[]):
        super(CONV_LSTM_AE, self).__init__()

        self.input_dims = input_dims
        self.conv_enc_dim = sum(input_dims)

        self.conv_ae = CONV_AE(
            input_dims,
            self.conv_enc_dim,
            kernel,
            stride,
            h_channels=h_conv_channels
        )
        self.lstm_ae = LSTM_AE(
            self.conv_enc_dim,
            encoding_dim,
            h_lstm_channels
        )

    def encoder(self, x):
        n_elements, encodings = x.shape[0], []
        for i in range(n_elements):
            element = x[i].unsqueeze(0).unsqueeze(0)
            encodings.append(self.conv_ae.encoder(element))

        return self.lstm_ae.encoder(torch.stack(encodings))

    def decoder(self, x, seq_len):
        encodings = self.lstm_ae.decoder(x, seq_len)
        decodings = []
        for i in range(seq_len):
            decodings.append(torch.squeeze(self.conv_ae.decoder(encodings[i])))

        return torch.stack(decodings)

    def forward(self, x):
        seq_len = x.shape[0]
        x = torchtranspose(x, 1, 2)
        x = self.encoder(x)
        x = self.decoder(x, seq_len)
        x = torchtranspose(x, 1, 2)
        return x
