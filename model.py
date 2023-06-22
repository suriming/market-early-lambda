import torch.nn as nn

class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LSTMEncoderDecoder, self).__init__()

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

        self.latent_dim = latent_dim

        self.fc_encode = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        encoded = self.fc_encode(hidden[-1])
        decoded, _ = self.decoder(self.fc_decode(encoded).unsqueeze(1).repeat(1, x.size(1), 1))
        return decoded, encoded