import pytorch_lightning as pl
from torch import nn, Tensor

from framework.model.utils.transformer_module import Transformer, LinearEmbedding
from framework.model.utils.position_embed import PositionalEncoding


class TransformerDecoder(pl.LightningModule):
    def __init__(self, latent_dim: int,
                 num_heads: int,
                 num_layers: int,
                 quant_factor: int,
                 intermediate_size: int,
                 nfeats: int,  # passed trough datamodule
                 is_audio=False,
                 **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.expander = nn.ModuleList()

        if quant_factor == 0:
            self.expander.append(nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, 5, stride=1, padding=2, padding_mode='replicate'),
                nn.LeakyReLU(0.2, False),
                nn.InstanceNorm1d(latent_dim, affine=False)))
        else:   # this is not used
            self.expander.append(nn.Sequential(
                nn.ConvTranspose1d(latent_dim, latent_dim, 5, stride=2, padding=2,
                                   output_padding=1, padding_mode='zeros'),
                nn.LeakyReLU(0.2, True),
                nn.InstanceNorm1d(latent_dim, affine=False)))
            num_layers = quant_factor + 2 if is_audio else quant_factor
            for _ in range(1, num_layers):
                self.expander.append(nn.Sequential(
                    nn.Conv1d(latent_dim, latent_dim, 5, stride=1, padding=2, padding_mode='zeros'),
                    nn.LeakyReLU(0.2, True),
                    nn.InstanceNorm1d(latent_dim, affine=False)))
            
        self.decoder_transformer = Transformer(
            in_size=latent_dim,
            hidden_size=latent_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size)
        self.decoder_pos_embedding = PositionalEncoding(latent_dim)
        self.decoder_linear_embedding = LinearEmbedding(latent_dim, latent_dim)
        self.feature_mapping_reverse = nn.Linear(latent_dim, nfeats, bias=False)

    def forward(self, inputs: Tensor):
        dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
        for i, module in enumerate(self.expander):
            inputs = module(inputs.permute(0, 2, 1)).permute(0, 2, 1)
            if i > 0:   # this is not used
                inputs = inputs.repeat_interleave(2, dim=1)

        decoder_features = self.decoder_linear_embedding(inputs)
        decoder_features = self.decoder_pos_embedding(decoder_features)
        decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
        pred_recons = self.feature_mapping_reverse(decoder_features)

        return pred_recons
