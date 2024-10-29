# Anatomical Foundation Models for Brain MRIs


## Pretrained Models

```python

import torch
from torchvision import transforms
from anatcl import AnatCL

model = AnatCL(descriptor="global", fold=0, pretrained=True)
model = model.to("cuda")

transform = transforms.Compose([
        transforms.Lambda(lambda x: torch.from_numpy(x.copy()).float()),
        transforms.Normalize(mean=0.0, std=1.0)
])

# Volumes should be 121x128x121 preprocessed with cat12 toolbox (vbm)
dataset = Dataset(transform=transform, ...)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False,
                                         num_workers=8, persistent_workers=True)

model.eval()
for (image, label) in dataloader:
    image = image.to("cuda")
    output = model(image)
    
    # Do something with the output
```

## Training Code

Coming soon

## Testing code

Coming soon