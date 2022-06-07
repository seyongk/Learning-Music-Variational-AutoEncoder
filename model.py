import torch.cuda
from torch import nn
from torch.distributions import OneHotCategorical
from loss import kl_divergence
import torch


class MusicVAE(nn.Module):
    def __init__(self):
        super(MusicVAE, self).__init__()
        self.encoder = Encoder()
        self.conductor = Conductor()
        self.decoder = Decoder()

        self.cache = None
        self._beta = nn.Parameter(torch.ones(size=(1,)), requires_grad=True)
        self.bce = nn.BCELoss()
        self.sample_hidden_state = None
        self.eta = 10.0

    def forward(self, x: torch.tensor, step_size: int, verbose: int=0):
        outputs = []
        batch_size = x.shape[0] // step_size
        loss = 0

        for n in range(step_size):
            input_seq = x[n * batch_size: (n + 1) * batch_size]
            if input_seq.shape[0] < batch_size:
                break
            initial_state_of_conductor, mu, log_var = self.encoder.forward(input_seq)
        self.cache = mu, log_var

        context = None
        for n in range(step_size):
            input_seq = x[n * batch_size: (n + 1) * batch_size]

            if context is None:
                context, (hidden_state, _) = self.conductor(initial_state_of_conductor)
            else:
                context, (hidden_state, _) = self.conductor(hidden_state)

            if context.shape[0] < batch_size:
                break

            probs = self.decoder(context)
            outputs.append(probs)
            recon_loss = self.bce(probs, input_seq)
            kl_loss = kl_divergence(mu, log_var)

            if verbose:
                print(f'{recon_loss = }, {kl_loss = }')
            loss += self.eta * recon_loss + self._beta * kl_loss

        return outputs, loss

    def sample(self):
        if self.sample_hidden_state is None:
            mu, log_var = self.cache
            initial_state_of_conductor = self.encoder.sample(mu, log_var)
            context, (self.sample_hidden_state, _) = self.conductor(initial_state_of_conductor)
        else:
            context, (self.sample_hidden_state, _) = self.conductor(self.sample_hidden_state)
        decoded_probs = self.decoder(context)
        return OneHotCategorical(decoded_probs)

    def initialize_sampler(self):
        self.sample_hidden_state = None

    @property
    def beta(self):
        return self._beta
    
    @beta.setter
    def beta(self, value):
        self._beta = value


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # bidirectional=True -> double output size
        self.lstm_1 = nn.LSTM(input_size=256, hidden_size=1024, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=2048, hidden_size=1024, bidirectional=True)

        self.fc_mu = nn.Linear(2048, 1024)
        self.fc_var = nn.Linear(2048, 1024)
        self.fc_z_emb = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh()
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        output, _ = self.lstm_1(x)
        output, _ = self.lstm_2(output)

        mu = self.fc_mu(output)
        log_var = self.fc_var(output)

        z_emb, _ = self.reparameterize(mu, log_var)
        initial_state_of_conductor = self.fc_z_emb(z_emb)

        return initial_state_of_conductor, mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.mul(log_var, 0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return mu + torch.mul(std, eps), std

    def sample(self, mu, log_var):
        z_emb, _ = self.reparameterize(mu, log_var)
        return self.fc_z_emb(z_emb)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=512, hidden_size=1024)
        self.lstm_2 = nn.LSTM(input_size=1024, hidden_size=256)
        self.softmax = nn.Softmax()

    def forward(self, x):
        out, _ = self.lstm_1(x)
        out, _ = self.lstm_2(out)
        return self.softmax(out)


class Conductor(nn.Module):
    def __init__(self):
        super(Conductor, self).__init__()
        self.conductor_1 = nn.LSTM(input_size=512, hidden_size=1024)
        self.conductor_2 = nn.LSTM(input_size=1024, hidden_size=512)

    def forward(self, x):
        output, _ = self.conductor_1(x)
        return self.conductor_2(output)
