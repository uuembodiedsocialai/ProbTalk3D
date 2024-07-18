from typing import List, Union, Tuple
import pytorch_lightning as pl
from torch import Tensor
import os
import logging
from framework.data.tools.collate import lengths_to_mask


logger = logging.getLogger(__name__)


class HuBERT(pl.LightningModule):
    def __init__(self, modelpath_processor: str,
                 modelpath_audiomodel: str,
                 finetune: bool = True,
                 **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        from transformers import Wav2Vec2Processor
        from transformers import logging
        from deps.hubert.modeling_hubert import HubertModel
        logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Processor: HuBERT uses the processor of Wav2Vec 2.0
        self.processor = Wav2Vec2Processor.from_pretrained(modelpath_processor)

        # Audio model:
        self.audiomodel = HubertModel.from_pretrained(modelpath_audiomodel)
        if finetune:     # Finetune the model
            self.audiomodel.feature_extractor._freeze_parameters()
            frozen_layers = [0, 1]
            for name, param in self.audiomodel.named_parameters():
                if name.startswith("feature_projection"):
                    param.requires_grad = False
                if name.startswith("encoder.layers"):
                    layer = int(name.split(".")[2])
                    if layer in frozen_layers:
                        param.requires_grad = False
        else:           # Freeze the model
            for name, param in self.audiomodel.named_parameters():
                param.requires_grad = False

        # Configure the model
        self.audio_encoded_dim = self.audiomodel.encoder.config.hidden_size

    def forward(self, audio: List[float],
                return_mask: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        audio_batch = []
        lengths = []
        for seq in audio:
            encoded_inputs = self.processor(seq, return_tensors="pt", padding=True)
            output = self.audiomodel(**encoded_inputs.to(self.audiomodel.device))   # [1, T, 768]
            audio_batch.append(output.last_hidden_state)
            lengths.append(output.last_hidden_state.shape[1])

        mask = lengths_to_mask(lengths, self.device)

        if not return_mask:
            return audio_batch
        return audio_batch, mask

