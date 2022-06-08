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

        # Hierarchcial Decoder의 구성 요소인 conductor 클래스의 인스턴스를 생성
        self.conductor = Conductor()
        self.decoder = Decoder()

        self.fc_mu = nn.Linear(1024, 256)
        self.fc_var = nn.Linear(1024, 256)
        self.fc_z_emb = nn.Sequential(nn.Linear(256, 512), nn.Tanh())

        self.fc_conductor = nn.Sequential(nn.Linear(512, 1024), nn.Tanh())

        self.bce = nn.BCELoss(reduction="sum")

        # Reparameterization에 사용되는 beta 파라미터
        self.max_beta = torch.Tensor([0.2]).to(
            self.device
        )  # Maximum KL cost weight, or cost if not annealing. small data: 0.2, big data: 0.5
        self.beta_rate = torch.Tensor([0.99999]).to(
            self.device
        )  # Exponential rate at which to anneal KL cost.
        self.global_step = 1
        self.beta = (
            (1.0 - torch.pow(self.beta_rate, self.global_step)) * self.max_beta
        ).to(self.device)

        # KL divergence에 Threshold로서 적용되어, Reconstruction loss에 좀 더 가중치를 주는 역할을 함
        self.free_bits = torch.Tensor([48.0]).to(self.device)

        # 모델 학습 후 sampling 시, 첫번째 iteration에만 Latent vector를 추출하기 위해 초기값을 None으로 설정
        self.sample_hidden_state = None
        self.cache = None

    def forward(self, x: torch.tensor, step_size: int, verbose: int = 0) -> float:

        self.global_step += 1

        # 전체 데이터(x)를 받아와서, forward 메소드 내부에서 batch 학습을 진행
        batch_size = x.shape[0] // step_size
        loss = 0

        for n_step in range(step_size):
            input_seq = x[n_step * batch_size : (n_step + 1) * batch_size]

            # input_seq의 크기가 나누어 떨어지지 않는 경우를 위해, batch_size 보다 작은 크기를 가진 경우 중단
            if input_seq.shape[0] < batch_size:
                break

            # 전체 데이터를 모두 활용해서 Encoder를 학습함
            output, _ = self.encoder.forward(input_seq)

        mu = self.fc_mu(output)
        log_var = self.fc_var(output)

        # 학습이 끝난 후 sampling 시에 사용하기 위해 mu, log_var을 저장
        self.cache = mu, log_var

        kl_div = kl_divergence(mu, log_var)
        kl_loss = torch.max(
            torch.abs(kl_div - self.free_bits), torch.Tensor([0]).to(self.device)
        )

        z_emb, _ = self.reparameterize(mu, log_var)
        initial_state_of_conductor = self.fc_z_emb(z_emb)

        for n_step in range(step_size):
            if n_step == 0:
                context, (hidden_state, _) = self.conductor(initial_state_of_conductor)
            else:
                context, (hidden_state, _) = self.conductor(hidden_state)

            if context.shape[0] < batch_size:
                break

            initial_state_of_decoder = self.fc_conductor(context)
            probs = self.decoder(initial_state_of_decoder)

            input_seq = x[n_step * batch_size : (n_step + 1) * batch_size]
            recon_loss = self.bce(probs, input_seq)

            if verbose:
                print(
                    f"Reconstruction loss: {recon_loss}, KL divergence loss: {kl_loss.item()}"
                )

            loss += recon_loss + self.beta * kl_loss

        return loss

    def reparameterize(
        self, mu: torch.tensor, log_var: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        # mean과 log variance의 분포로부터 latent vector를 추출
        std = torch.mul(log_var, 0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return mu + torch.mul(std, eps), std

    def sample(self) -> torch.tensor:
        # 모델 학습 후 sampling 시, 첫번째 iteration에만 Latent vector를 추출해서,
        # conductor의 인자로 쓰고, 이후의 sampling 시에는 hidden_state를 인자로 적용
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
        # sampling을 한 후에 새롭게 sampling을 하려 할 경우에 초기화함
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
