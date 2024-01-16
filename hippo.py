"""
Implementation of HiPPO RNN Model
"""

from typing import Optional, Tuple
import torch
import torch.linalg

from rnn import init_hidden


def get_state_input_matrix(N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    q = torch.arange(0, N, 1, dtype=torch.float64)
    col, row = torch.meshgrid(q, q, indexing="xy")
    r = 2 * q + 1
    M = -(torch.where(row >= col, r, 0) - torch.diag(q))
    T = torch.sqrt(torch.diag(r))
    A = torch.matmul(T, torch.matmul(M, torch.linalg.inv(T)))
    B = torch.diag(T)[:, None]
    return A, B


def legs_discretization(A: torch.Tensor, B: torch.Tensor, K: int, alpha: int = 0.5):
    """
    LegS formulation is that of a LTV (Linear Time Varying) system
    so we need to write our own discretization.

    Inspired by https://github.com/scipy/scipy/blob/v1.11.4/scipy/signal/_lti_conversion.py#L335-L533
    """
    ima = torch.eye(A.shape[0]) - alpha * A / (K + 1)
    dA = torch.linalg.solve_triangular(
        ima, torch.eye(A.shape[0]) + (1 - alpha) * 1 / K * A, upper=False
    )
    dB = torch.linalg.solve_triangular(
        ima, ((1 - alpha) * 1 / K + alpha * 1 / (K + 1)) * B, upper=False
    )
    return dA, dB


class SimpleHiPPOLegS(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        max_timesteps: int = 1024,
        alpha: float = 0.5,  # bilinear discretization
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_timesteps = max_timesteps
        A, B = get_state_input_matrix(self.hidden_size)
        A_stacked = torch.empty(
            (self.max_timesteps, self.hidden_size, self.hidden_size), dtype=A.dtype
        )
        B_stacked = torch.empty(
            (self.max_timesteps, self.hidden_size, 1), dtype=B.dtype
        )
        for k in range(max_timesteps):
            da, db = legs_discretization(A, B, k + 1, alpha)
            A_stacked[k] = da
            B_stacked[k] = db

        A_stacked -= torch.eye(self.hidden_size)

        self.register_buffer(
            "A", A_stacked.float()
        )  # (max_time_steps, hidden_size, hidden_size)
        self.register_buffer("B", B_stacked.float())  # (max_time_steps, hidden_size, 1)

        self.hidden_u = torch.nn.Linear(
            self.input_size + self.hidden_size, 1, bias=True
        )

        self.hidden_h = torch.nn.Linear(
            self.input_size + self.hidden_size, self.hidden_size, bias=True
        )

        self.hidden_g = torch.nn.Linear(
            self.input_size + self.hidden_size, self.hidden_size, bias=True
        )
        self.init_hidden()

    def init_hidden(self):
        init_hidden(self.hidden_g, self.input_size)
        init_hidden(self.hidden_h, self.input_size)
        init_hidden(self.hidden_u, self.input_size)

    def update_memory(self, m: torch.Tensor, u: torch.Tensor, k: int):
        if k > self.max_timesteps:
            k = self.max_timesteps
        return (
            m
            + torch.nn.functional.linear(m, self.A[k])
            + torch.nn.functional.linear(u, self.B[k])
        )

    def forward(
        self,
        inputs: torch.Tensor,
        H_C_t: Optional[Tuple[torch.Tensor, torch.Tensor, int]],
    ):
        if H_C_t is None:
            t = 0
            hidden = torch.zeros(
                (inputs.shape[1], self.hidden_size),
                device=inputs.device,
            )
            memory = torch.zeros(
                (inputs.shape[1], self.hidden_size),
                device=inputs.device,
            )
        else:
            hidden, memory, t = H_C_t

        outputs = []
        for input in inputs:
            U = self.hidden_u(torch.cat([input, hidden], 1))
            memory = self.update_memory(memory, U, t)
            H = torch.nn.functional.tanh(self.hidden_h(torch.cat([input, memory], 1)))
            G = torch.nn.functional.sigmoid(
                self.hidden_g(torch.cat([input, memory], 1))
            )
            hidden = G * hidden + (1 - G) * H
            outputs.append(hidden)
            t = t + 1
        return torch.stack(outputs, 0), (hidden, memory, t)


if __name__ == "__main__":
    get_state_input_matrix(10)
