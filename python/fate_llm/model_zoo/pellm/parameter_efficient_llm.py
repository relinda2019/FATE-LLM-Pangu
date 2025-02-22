#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import peft
import torch
from peft import PeftModel
from transformers import AutoConfig
from transformers import AutoModel
from transformers.configuration_utils import PretrainedConfig
from federatedml.util import LOGGER


AVAILABLE_PEFT_CONFIG = list(
    filter(
        lambda peft_type: peft_type.endswith("Config"), dir(peft)
    )
)


class PELLM(torch.nn.Module):

    config_class: PretrainedConfig = None
    enable_save_pretrained: bool = True
    model_loader = None

    def __init__(self, config: dict = None,
                 pretrained_path: str = None,
                 peft_type: str = None,
                 peft_config: dict = None,
                 **kwargs
                 ) -> None:

        super().__init__()
        self._pe_lm: PeftModel = None
        self.config = config
        self.config_path = pretrained_path
        self.peft_type = peft_type
        self.peft_config = peft_config

        assert self.config_path is not None or self.config is not None, \
            "At least one of config_path and config must be set."
        self._init_pelm(**kwargs)

    def _init_pelm(self, **kwargs):
        self.init_lm_with_peft(**kwargs)
        self.model_summary()

    def init_lm_with_peft(self, **kwargs):
        self.init_config(**kwargs)
        self.init_base_lm()
        self.add_peft()

    def init_config(self, **kwargs):
        if self.config_path is not None:
            self.config = AutoConfig.from_pretrained(self.config_path)
        elif self.config is not None and self.config_class is not None:
            self.config = self.config_class().from_dict(self.config)
        else:
            raise ValueError(
                'config_path to pretrained model folder and model config dict cannot be None at the same time, '
                'you need to specify one of them')

        if kwargs:
            self.config.update(kwargs)

    def init_base_lm(self, **kwargs):
        model_loader = self.model_loader if self.model_loader is not None else AutoModel
        if self.config is not None:
            self._pe_lm = model_loader.from_pretrained(
                self.config_path, config=self.config, **kwargs)
        elif self.config_path is not None:
            self._pe_lm = model_loader.from_pretrained(
                self.config_path, **kwargs)
        else:
            raise ValueError(
                'config_path to pretrained model folder cannot be None')

    def add_peft(self):
        assert self.peft_type in AVAILABLE_PEFT_CONFIG, 'peft name {} not in availabe config {}'.format(
            self.peft_type, AVAILABLE_PEFT_CONFIG)

        if self.peft_config is None:
            peft_config = getattr(peft, self.peft_type)()
        else:
            peft_config = getattr(peft, self.peft_type)(**self.peft_config)

        self._pe_lm = peft.get_peft_model(self._pe_lm, peft_config)

    def model_summary(self):
        try:
            summary = self._pe_lm.print_trainable_parameters()

            LOGGER.debug('PELLM model summary: \n{}'.format(summary))
        except BaseException:
            pass

    def _get_trainable_parameters(self):
        trainable = []
        for n, p in self._pe_lm.named_parameters():
            if p.requires_grad:
                trainable.append(p)
        return trainable

    def forward(self, tokenized_data: dict):
        return self._pe_lm(**tokenized_data)

    def save_pretrained(self, path):
        if not self.enable_save_pretrained:
            raise ValueError(
                "To save trainable parameters only, set enable_save_pretrained=True in your model")

        from pathlib import Path

        state_dict = {
            k: p.to("cpu") for k,
            p in self._pe_lm.named_parameters() if p.requires_grad}
        Path.mkdir(Path(path), exist_ok=True)
        torch.save(state_dict, Path(path).joinpath("adapter_model.bin"))


class AutoPELLM(PELLM):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
