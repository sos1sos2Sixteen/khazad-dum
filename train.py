import torch
import torchvision
from model import DiffusionUnet
import pytorch_lightning as pl


mnistdata = torchvision.datasets.MNIST(
    'datasets/mnist', 
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.13066047], [0.30150425]),
        torchvision.transforms.Lambda(lambda x: x.squeeze(0))
    ])
)
data_loader = torch.utils.data.DataLoader(
    mnistdata,
    batch_size=128,
    shuffle=True,
    num_workers=0)


trainer = pl.Trainer(max_epochs=10000, accelerator='gpu', devices='1')
diff = DiffusionUnet(1000)

trainer.fit(diff, data_loader)