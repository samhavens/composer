# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Monitor stuff rate during training."""

from composer import Callback, Event, Logger, State

__all__ = ['SamCallback']

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

    The stuff is 

    +---------------------------------------------+---------------------------------------+
    | Key                                         | Logged data                           |
    +=============================================+=======================================+
    |                                             | Learning rate for each optimizer and  |
    | `stuff``                                    | parameter group for that optimizer is |
    |                                             | logged to a separate key.             |
    +---------------------------------------------+---------------------------------------+
    """
    count = 0

    def __init__(self) -> None:
        pass

    def run_event(self, event: Event, state: State, logger: Logger):
        if event == Event.BATCH_END:
            self.count += 1
            if self.count % 100 == 0:
                print('yo')