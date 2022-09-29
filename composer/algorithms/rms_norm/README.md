# 🦷 Root Mean Square Layer Norm

[\[How to Use\]](#how-to-use) - [\[Suggested Hyperparameters\]](#suggested-hyperparameters) - [\[Technical Details\]](#technical-details) - [\[Attribution\]](#attribution)

 `Natural Language Processing`

Root Mean Square Layer Norm, or RMSNorm, regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efficient than LayerNorm.


## How to Use

### Functional Interface

```python
# Run the RMSNorm algorithm directly on the model using the Composer functional API

import torch
import torch.nn.functional as F
import composer.functional as cf

def training_loop(model, train_loader):
    opt = torch.optim.Adam(model.parameters())

    # only need to pass in opt if apply_rms_norm is used after
    # optimizer creation; otherwise only the model needs to be passed in
    cf.apply_rms_norm(
        model,
        optimizers=opt
    )

    loss_fn = F.cross_entropy
    model.train()

    for epoch in range(1):
        for X, y in train_loader:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
```

### Composer Trainer

<!--pytest.mark.gpu-->
<!--
```python
from torch.utils.data import DataLoader
from tests.common import RandomClassificationDataset, SimpleModel

model = SimpleModel()
train_dataloader = DataLoader(RandomClassificationDataset())
eval_dataloader = DataLoader(RandomClassificationDataset())
```
-->
<!--pytest-codeblocks:cont-->
```python
# Instantiate the algorithm and pass it into the Trainer
# The trainer will automatically run it at the appropriate point in the training loop

from composer.algorithms import RMSNorm
from composer.trainer import Trainer

rmsn = RMSNorm()

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration='10ep',
    algorithms=[rmsn]
)

trainer.fit()
```

## Suggested Hyperparameters

The only hyperparameter is `eps`, a value added to the denominator for numerical stability. It's default value is 1e-8, which we observed working well.

## Technical Details

The Composer implementation of RMSNorm uses model surgery to replace `LayerNorm` layers with `RMSNorm` layers.

...NOTES FROM ACTUALLY USING IT...

> As a general rule, composing regularization methods may lead to diminishing returns in quality improvements. Root Mean Square Layer Norm is one such regularization method.


## Attribution

[*Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467) by Biao Zhang and Rico Sennrich. Published in NeurIPS in 2019.

[*Do Transformer Modifications Transfer Across Implementations and Applications?*](https://arxiv.org/abs/2102.11972) by Sharan Narang, Hyung Won Chung, Yi Tay, William Fedus, Thibault Fevry, Michael Matena, Karishma Malkan, Noah Fiedel, Noam Shazeer, Zhenzhong Lan, Yanqi Zhou, Wei Li, Nan Ding, Jake Marcus, Adam Roberts, and Colin Raffel. Published in EMNLP in 2021.

*The Composer implementation of this method and the accompanying documentation were produced by Sam Havens at MosaicML.*
