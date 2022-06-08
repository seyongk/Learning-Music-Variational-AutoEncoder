from typing import List, Tuple

import torch
import torch.cuda
from torch import nn
from torch.distributions import OneHotCategorical

from loss import kl_divergence


class MusicVAE(nn.Module):
    def __init__(self):
        super(MusicVAE, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.encoder = Encoder()
        self.conductor = Conductor()
        self.decoder = Decoder()

        self.fc_mu = nn.Linear(1024, 256)
        self.fc_var = nn.Linear(1024, 256)
        self.fc_z_emb = nn.Sequential(nn.Linear(256, 512), nn.Tanh())

        self.fc_conductor = nn.Sequential(
                    nn.Linear(512, 1024),
                    nn.Tanh()
        )

        self.bce = nn.BCELoss(reduction="sum")

        self.max_beta = torch.Tensor([0.2]).to(self.device)  # Maximum KL cost weight, or cost if not annealing. small data: 0.2, big data: 0.5
        self.beta_rate = torch.Tensor([0.99999]).to(self.device)  # Exponential rate at which to anneal KL cost.
        self.global_step = 1
        self.beta = ((1.0 - torch.pow(self.beta_rate, self.global_step)) * self.max_beta).to(self.device)
        self.free_bits = torch.Tensor([48.0]).to(self.device)

        self.sample_hidden_state = None
        self.cache = None

    def forward(
        self, x: torch.tensor, step_size: int, verbose: int = 0
    ) -> Tuple[List[torch.tensor], float]:

        self.global_step += 1
        batch_size = x.shape[0] // step_size
        loss = 0

        for iter in range(step_size):
            input_seq = x[iter * batch_size : (iter + 1) * batch_size]

            if input_seq.shape[0] < batch_size:
                break

            output, _ = self.encoder.forward(input_seq)

        mu = self.fc_mu(output)
        log_var = self.fc_var(output)
        self.cache = mu, log_var
        kl_loss = torch.max(torch.abs(kl_divergence(mu, log_var) - self.free_bits), torch.Tensor([0]).to(self.device))

        z_emb, _ = self.reparameterize(mu, log_var)
        initial_state_of_conductor = self.fc_z_emb(z_emb)

        for iter in range(step_size):
            if iter == 0:
                context, (hidden_state, _) = self.conductor(initial_state_of_conductor)
            else:
                context, (hidden_state, _) = self.conductor(hidden_state)

            if context.shape[0] < batch_size:
                break

            initial_state_of_decoder = self.fc_conductor(context)
            probs = self.decoder(initial_state_of_decoder)

            input_seq = x[iter * batch_size : (iter + 1) * batch_size]
            recon_loss = self.bce(probs, input_seq)

            if verbose:
                print(f"Reconstruction loss: {recon_loss}, KL divergence loss: {kl_loss.item()}")

            loss += recon_loss + self.beta * kl_loss

        return loss

    def reparameterize(
        self, mu: torch.tensor, log_var: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        std = torch.mul(log_var, 0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return mu + torch.mul(std, eps), std

    def sample(self) -> torch.tensor:
        if self.sample_hidden_state is None:
            mu, log_var = self.cache
            z_emb, _ = self.reparameterize(mu, log_var)
            initial_state_of_conductor = self.fc_z_emb(z_emb)
            context, (self.sample_hidden_state, _) = self.conductor(
                initial_state_of_conductor
            )
        else:
            context, (self.sample_hidden_state, _) = self.conductor(
                self.sample_hidden_state
            )
        initial_state_of_decoder = self.fc_conductor(context)
        decoded_probs = self.decoder(initial_state_of_decoder)
        return OneHotCategorical(decoded_probs)

    def initialize_sampler(self) -> None:
        self.sample_hidden_state = None


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # bidirectional=True -> double output size
        self.lstm_1 = nn.LSTM(input_size=256, hidden_size=1024, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=2048, hidden_size=512, bidirectional=True)

    def forward(
        self, x: torch.tensor
    ) -> Tuple[torch.tensor, Tuple[torch.tensor, torch.tensor]]:
        output, _ = self.lstm_1(x)
        return self.lstm_2(output)


class Conductor(nn.Module):
    def __init__(self):
        super(Conductor, self).__init__()
        self.conductor_1 = nn.LSTM(input_size=512, hidden_size=1024)
        self.conductor_2 = nn.LSTM(input_size=1024, hidden_size=512)

    def forward(self, x: torch.tensor) -> torch.tensor:
        output, _ = self.conductor_1(x)
        return self.conductor_2(output)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lstm_1 = nn.LSTM(input_size=1024, hidden_size=1024)
        self.lstm_2 = nn.LSTM(input_size=1024, hidden_size=256)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.tensor) -> torch.tensor:
        out, _ = self.lstm_1(x)
        out, _ = self.lstm_2(out)
        return self.softmax(out)
