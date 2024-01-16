from typing import Optional
import torch


def init_hidden(hidden: torch.nn.Linear, input_size: int) -> None:
    torch.nn.init.xavier_normal_(hidden.weight[:, :input_size])
    torch.nn.init.xavier_normal_(hidden.weight[:, input_size:])
    torch.nn.init.zeros_(hidden.bias)


class SimpleRNN(torch.nn.Module):
    """
    Implementation based on the Dive into Deep Learning book
    https://d2l.ai/chapter_recurrent-neural-networks/rnn.html
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden = torch.nn.Linear(input_size + hidden_size, hidden_size, bias=True)
        self.init_hidden()

    def init_hidden(self):
        init_hidden(self.hidden, self.input_size)

    def forward(self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        if hidden is None:
            hidden = torch.zeros(
                (inputs.shape[1], self.hidden_size), device=inputs.device
            )
        output = []
        for input in inputs:
            hidden = torch.nn.functional.tanh(
                self.hidden(torch.cat([input, hidden], 1))
            )
            output.append(hidden)
        return torch.stack(output, 0), hidden


class SimpleGRU(torch.nn.Module):
    """
    Implementation based on the Dive into Deep Learning book
    https://d2l.ai/chapter_recurrent-modern/gru.html
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_r = torch.nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )
        self.hidden_z = torch.nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )
        self.hidden_h = torch.nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )
        self.init_hidden()

    def init_hidden(self):
        init_hidden(self.hidden_r, self.input_size)
        init_hidden(self.hidden_z, self.input_size)
        init_hidden(self.hidden_h, self.input_size)

    def forward(self, inputs: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        if hidden is None:
            hidden = torch.zeros(
                (inputs.shape[1], self.hidden_size), device=inputs.device
            )
        outputs = []
        for input in inputs:
            R = torch.nn.functional.sigmoid(
                self.hidden_r(torch.cat([input, hidden], 1))
            )
            Z = torch.nn.functional.sigmoid(
                self.hidden_z(torch.cat([input, hidden], 1))
            )
            H = torch.nn.functional.tanh(
                self.hidden_h(torch.cat([input, R * hidden], 1))
            )
            hidden = Z * hidden + (1 - Z) * H
            outputs.append(hidden)
        return torch.stack(outputs, 0), hidden


class SimpleLSTM(torch.nn.Module):
    """
    Implementation based on the Dive into Deep Learning book
    https://d2l.ai/chapter_recurrent-modern/lstm.html
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_i = torch.nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )
        self.hidden_f = torch.nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )
        self.hidden_o = torch.nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )
        self.hidden_c = torch.nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )
        self.init_hidden()

    def init_hidden(self):
        init_hidden(self.hidden_i, self.input_size)
        init_hidden(self.hidden_f, self.input_size)
        init_hidden(self.hidden_o, self.input_size)
        init_hidden(self.hidden_c, self.input_size)

    def forward(self, inputs: torch.Tensor, H_C: Optional[torch.Tensor] = None):
        if H_C is None:
            # Initial state with shape: (batch_size, num_hiddens)
            hidden = torch.zeros(
                (inputs.shape[1], self.hidden_size), device=inputs.device
            )
            C = torch.zeros((inputs.shape[1], self.hidden_size), device=inputs.device)
        else:
            hidden, C = H_C
        outputs = []
        for input in inputs:
            I = torch.nn.functional.sigmoid(
                self.hidden_i(torch.cat([input, hidden], 1))
            )
            F = torch.nn.functional.sigmoid(
                self.hidden_f(torch.cat([input, hidden], 1))
            )
            O = torch.nn.functional.sigmoid(
                self.hidden_o(torch.cat([input, hidden], 1))
            )
            C_tilde = torch.nn.functional.tanh(
                self.hidden_c(torch.cat([input, hidden], 1))
            )
            C = F * C + I * C_tilde
            hidden = O * torch.nn.functional.tanh(C)
            outputs.append(hidden)
        return torch.stack(outputs, 0), (hidden, C)


def clip_gradients(grad_clip_val, model: torch.nn.Module):
    """
    Shamelessly lifted from here
    https://d2l.ai/chapter_recurrent-neural-networks/rnn-scratch.html
    """
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm


class RNNClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        rnn_class: torch.nn.Module = SimpleRNN,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_class = rnn_class
        self.hidden_size = hidden_size
        self.layers = torch.nn.ModuleList(
            [
                rnn_class(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )
        self.linear = torch.nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, inputs: torch.Tensor):
        Hs = [None] * self.num_layers
        outputs = inputs
        for i, layer in enumerate(self.layers):
            outputs, Hs[i] = layer(outputs, Hs[i])
        return self.linear(outputs[-1])
