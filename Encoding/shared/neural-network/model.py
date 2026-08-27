import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity,
}

class Autoencoder(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 25,
        hidden_1_dim: int = 16,
        hidden_2_dim: int = 8,
        latent_dim: int = 3,
        hidden_activation: str = "relu",
        latent_activation: str = "tanh",
        output_activation: str = "none",
        dropout_1: float = 0.0,
        dropout_2: float = 0.0,
    ):
        super().__init__()

        if hidden_activation not in ACTIVATIONS:
            raise ValueError(f"Unknown hidden_activation: {hidden_activation}")
        if latent_activation not in ACTIVATIONS:
            raise ValueError(f"Unknown latent_activation: {latent_activation}")
        if output_activation not in ACTIVATIONS:
            raise ValueError(f"Unknown output_activation: {output_activation}")

        hidden_act = ACTIVATIONS[hidden_activation]
        latent_act = ACTIVATIONS[latent_activation]
        output_act = ACTIVATIONS[output_activation]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1_dim),
            hidden_act(),
            nn.Dropout(p=dropout_1),
            nn.Linear(hidden_1_dim, hidden_2_dim),
            hidden_act(),
            nn.Dropout(p=dropout_2),
            nn.Linear(hidden_2_dim, latent_dim),
            latent_act(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_2_dim),
            hidden_act(),
            nn.Linear(hidden_2_dim, hidden_1_dim),
            hidden_act(),
            nn.Linear(hidden_1_dim, input_dim),
            output_act(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z
    