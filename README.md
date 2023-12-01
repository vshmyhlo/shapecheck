# `shapecheck`

Small utility to check shapes of your arrays/tensors

## Examples

### Check shapes for batch matrix multiplication

```python
import torch
from shapecheck import shapecheck

def some_code_that_does_batch_matmul(a: torch.Tensor, b: torch.Tensor):
    s = shapecheck()

    # N matrices of shape (A, B)
    s.check("NAB", a)

    # N matrices of shape (B, C)
    s.check("NBC", b)

    c = torch.bmm(a, b)

    # N matrices of shape (A, C)
    s.check("NAC", c)

    return c
```

### Check shapes for contrastive training on images

```python
import torch
from torchvision.models import resnet18
import torch.nn as nn
from shapecheck import shapecheck

# batch of 10 pairs of (3, 128, 256) images for contrastive training
image_pairs = torch.zeros(10, 2, 3, 128, 256)

s = shapecheck()
s.check("BNCHW", image_pairs)
assert s["N"] == 2  # check those are pairs indeed
assert s["C"] == 3  # check images have 3 channels

# flatten to feed into model
image_pairs = image_pairs.flatten(0, 1)

# forward pass
model = resnet18()
model.fc = nn.Identity()
outputs = model(image_pairs)

# unflatten back into pairs
outputs = outputs.unflatten(0, (s["B"], s["N"]))

# batch of 10 pairs of (512) vectors
s.check("BNc", outputs)
assert s["c"] == 512
```
