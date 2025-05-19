# Anatomical Foundation Models for Brain MRIs


## Installing

```
pip3 install git+https://github.com/carloalbertobarbano/AnatCL.git
```

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

## Citing

If you find our models useful please do not forget to cite this work as

```bibtex
@article{barbano2024anatomical,
  title={Anatomical Foundation Models for Brain MRIs},
  author={Barbano, Carlo Alberto and Brunello, Matteo and Dufumier, Benoit and Grangetto, Marco},
  journal={arXiv preprint arXiv:2408.07079},
  year={2024}
}
```