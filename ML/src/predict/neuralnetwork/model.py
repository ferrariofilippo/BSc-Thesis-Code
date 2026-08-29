import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity,
}

class NeuralNetworkModel(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 59,
        hidden_1_dim: int = 32,
        hidden_2_dim: int = 32,
        hidden_3_dim: int = 32,
        output_dim: int = 1,
        hidden_activation: str = "relu",
        output_activation: str = "none",
    ):
        super().__init__()

        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"Unknown hidden_activation: {hidden_activation}")
        if output_activation not in ACTIVATIONS:
            raise ValueError(f"Unknown output_activation: {output_activation}")

        hidden_act = ACTIVATIONS[hidden_activation]
        output_act = ACTIVATIONS[output_activation]

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_1_dim),
            hidden_act(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_1_dim, hidden_2_dim),
            hidden_act(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_2_dim, hidden_3_dim),
            hidden_act(),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_3_dim, output_dim),
            # hidden_act(),
            # nn.Dropout(p=0.0),
            # nn.Linear(hidden_2_dim, output_dim),
            output_act(),
        )

    def forward(self, x):
        return self.model(x)
