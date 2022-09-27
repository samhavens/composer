# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor stuff rate during training."""

from composer import Callback, Event, Logger, State

from transformers import AutoTokenizer

import numpy as np
import torch

__all__ = ['SamCallback']

tok_name = "bert-base-uncased" # is there a WAY TO GET THIS FROM STATE OR SOMETHING?

tokenizer = AutoTokenizer.from_pretrained(tok_name)

class SamCallback(Callback):
    """Logs stuff

    This callback does stuff


    Example:
        .. doctest::

            >>> from composer import Trainer
            >>> from composer.callbacks import SamCallback
            >>> # constructing trainer object with this callback
            >>> trainer = Trainer(
            ...     model=model,
            ...     train_dataloader=train_dataloader,
            ...     eval_dataloader=eval_dataloader,
            ...     optimizers=optimizer,
            ...     max_duration="1ep",
            ...     callbacks=[SamCallback()],
            ... )

    The stuff is stuffy
    """

    def __init__(self) -> None:
        self.count = 0
        self.every = 100  # should be 100

    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.BATCH_END:
            self.count += 1
            if self.count % self.every == 0:
                print(f'Number of batches: {self.count}')
                iids = state.batch['input_ids'][0]
                print(f"INPUT: {tokenizer.decode(iids)}")
                # Find the location of [MASK] and extract its logits
                mask_token_index = torch.argwhere(iids == tokenizer.mask_token_id)
                # print(mask_token_index.shape)
                mask_token_logits = state.outputs.logits[0, mask_token_index, :]
                # Pick the [MASK] candidates with the highest logits
                top_ids = torch.argmax(mask_token_logits, dim=2)
                top_tokens = tokenizer.batch_decode(top_ids)
                print(f"top guesses for each [MASK]: {top_tokens}")
                out = ""
                for s, guess in zip(tokenizer.decode(iids).split(tokenizer.mask_token), top_tokens):
                    # tokenizer.decode does smart token-joining stuff, so I don't want to rewrite that, hence this split/zip thing
                    out += s + guess
                print(f"OUTPUT: {out}")
