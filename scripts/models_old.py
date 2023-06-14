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
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    batch_size = x.shape[0]
    #x = x.reshape((batch_size, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((batch_size, self.embedding_dim))
  
class Decoder(nn.Module):
  def __init__(self, seq_len, n_features=1, input_dim=64,):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):
    batch_size = x.shape[0]
    x = x.unsqueeze(1)
    x = x.repeat(1, self.seq_len, 1)
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    #x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
    return self.output_layer(x)
  
class LSTM_AE(nn.Module):
  def __init__(self, num_ifos, num_timesteps, BOTTLENECK, FACTOR):
    super(LSTM_AE, self).__init__()
    print("WARNING: This is LITERALLY Eric's model!!!")
    for i in range(50): 
       continue
       print("MINECRAFT MINECRAFT MINECRAFT")
    self.num_timesteps = num_timesteps
    self.num_ifos = num_ifos
    self.BOTTLENECK = BOTTLENECK
    #self.encoder = Encoder(seq_len=num_timesteps, n_features=num_ifos, embedding_dim=BOTTLENECK)
    #self.decoder = Decoder(seq_len=num_timesteps, n_features=num_ifos, input_dim=BOTTLENECK)
    self.encoder = Encoder(seq_len=num_ifos, n_features=num_timesteps, embedding_dim=BOTTLENECK)
    self.decoder = Decoder(seq_len=num_ifos, n_features=num_timesteps, input_dim=BOTTLENECK)
  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TransformerAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = torchtranspose(x, 1, 2)
        batch_size, seq_len, _ = x.size()
        # Encoder
        encoding = self.encoder(x.transpose(0, 1))
        # Decoder
        decoding = self.decoder(encoding, encoding)
        # Reshape for fully connected layer
        decoding = decoding.transpose(0, 1).contiguous().view(batch_size * seq_len, -1)
        # Fully connected layer
        output = self.fc(decoding).view(batch_size, seq_len, -1)
        #output = torchtranspose(decoding, 1, 2)
        #print(output.view(batch_size, seq_len, -1).shape)
        return torchtranspose(output, 1, 2)
  