import pytorch_lightning as pl
from torch import nn

from framework.model.utils.transformer_module import Transformer, LinearEmbedding
from framework.model.utils.position_embed import PositionalEncoding


class TransformerPredictor(pl.LightningModule):
    def __init__(self, latent_dim: int,
                 num_layers: int,
                 num_heads: int,
                 quant_factor: int,
                 intermediate_size: int,
                 # passed trough datamodule
                 audio_dim: int,
                 one_hot_dim: int,
                 **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.audio_feature_map = nn.Linear(audio_dim, latent_dim)               # 768*2->256
        self.style_embedding = nn.Linear(one_hot_dim, latent_dim, bias=False)   # 43->256

        if quant_factor == 0:
            layers = [nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, 5, stride=1, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(0.2, True),
                nn.InstanceNorm1d(latent_dim, affine=False))]
        else:   # this is not used
            layers = [nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, 5, stride=2, padding=2, padding_mode='zeros'),
                nn.LeakyReLU(0.2, True),
                nn.InstanceNorm1d(latent_dim, affine=False))]
            for _ in range(1, quant_factor):
                layers += [nn.Sequential(
                    nn.Conv1d(latent_dim, latent_dim, 5, stride=1, padding=2, padding_mode='zeros'),
                    nn.LeakyReLU(0.2, True),
                    nn.InstanceNorm1d(latent_dim, affine=False),
                    nn.MaxPool1d(2)
                )]

        self.squasher = nn.Sequential(*layers)
        self.encoder_transformer = Transformer(
            in_size=latent_dim,
            hidden_size=latent_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size)
        self.encoder_pos_embedding = PositionalEncoding(latent_dim, batch_first=True)
        self.encoder_linear_embedding = LinearEmbedding(latent_dim, latent_dim)

    def forward(self, audio, style_ont_hot):
        """
        audio: [B, T, 768*2]
        style_ont_hot: [B, 43]
        """
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        inputs = self.audio_feature_map(audio)                  # [B=1, T, 256]

        # style embedding
        style_ont_hot = self.style_embedding(style_ont_hot)     # [1, 256]
        inputs = inputs * style_ont_hot                         # [1, T, 256]

        inputs = self.squasher(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        encoder_features = self.encoder_linear_embedding(inputs)
        encoder_features = self.encoder_pos_embedding(encoder_features)
        encoder_features = self.encoder_transformer((encoder_features, dummy_mask))

        return encoder_features








