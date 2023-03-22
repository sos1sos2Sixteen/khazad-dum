
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt 
import pytorch_lightning as pl
from typing import Optional, Tuple
from unet.unet import UNet
from tqdm import tqdm

def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(15,5))
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    return fig

def batch_weight(weight, data): 
    '''
    result[i] = data[i] * weight[i]
    weight: (bcsz, )
    data: (bcsz, ...)
    '''
    x = data.permute(*torch.arange(data.ndim - 1, -1, -1))
    x = x * weight
    return x.permute(*torch.arange(x.ndim - 1, -1, -1))

class NoiseSchedule(): 
    def alpha(self, t: int) -> float: 
        raise NotImplementedError()

    def beta(self, t: int) -> float: 
        raise NotImplementedError()

    def alpha_bar(self, t: int) -> float: 
        raise NotImplementedError()

    def beta_tilde(self, t: int) -> float: 
        raise NotImplementedError()

class LinearNoiseSchedule(NoiseSchedule, nn.Module): 
    def __init__(self, beta0: float, beta1: float, T: int) -> None: 
        super().__init__()

        self.T = T
        _betas = torch.linspace(beta0, beta1, T, dtype=torch.float)
        _alpha_bars = torch.zeros_like(_betas)

        _alpha_bars[0] = 1 - _betas[0]
        for i in range(1, T): 
            _alpha_bars[i] = _alpha_bars[i-1] * (1 - _betas[i])
        
        self.register_buffer('_betas', _betas)
        self.register_buffer('_alpha_bars', _alpha_bars)
        
    def alpha(self, t: int) -> float:
        return 1 - self.beta(t)
    
    def beta(self, t: int) -> float: 
        return self._betas[t]

    def alpha_bar(self, t: int) -> float:
        return self._alpha_bars[t]


class DiffusionUnet(pl.LightningModule): 
    def __init__(self, T): 
        super().__init__()
        
        self.T = T
        imgsize = 784
        tebsize = 32
        self.znet = UNet(dim_emb=28 * 28)
        self.ns = LinearNoiseSchedule(1e-4, 0.01, T)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.znet.parameters(), lr=2e-4)
        schlr = torch.optim.lr_scheduler.StepLR(optim, 80)
        return [optim], [schlr]
    
    def training_step(self, batch, batch_idx): 
        '''
        batch: (bcsz, channel, h, w)
        '''
        x0, _ = batch 
        bcsz, *_ = x0.shape
        x0 = x0.unsqueeze(1)
        t = torch.randint(0, self.T-1, (bcsz, ), device=self.device)
        zt = torch.randn_like(x0, device=self.device)

        xt = self.diffuse(x0, t, zt)

        z_pred = self.znet(xt, t) 

        loss = F.mse_loss(z_pred, zt)
        
        self.log('loss', loss.item())

        return loss
    
    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()

        with torch.no_grad(): 
            r = self.sample_interval(torch.randn(10, 1, 28, 28,).to(self.device), (999, 1))
            # r = torch.randn(10, 1, 28, 28,).to(self.device)
            r = r[:, 0].cpu()
            fig = plot(r)
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            self.logger.experiment.add_image(
                'sampled', data, 
                dataformats='HWC', 
                global_step = self.global_step
            )

    
    def diffuse(self, x0: torch.Tensor, t: torch.Tensor, zt: Optional[torch.Tensor] = None) -> torch.Tensor: 
        if zt is None: 
            zt = torch.randn_like(x0, device=self.device) 
        return batch_weight(torch.sqrt(self.ns.alpha_bar(t)), x0) + batch_weight(torch.sqrt(1 - self.ns.alpha_bar(t)), zt)
    
    def sample_step(self, x_t2: torch.Tensor, t_idx: int, zt = Optional[torch.Tensor]) -> torch.Tensor: 
        assert 0 <= t_idx < self.T

        print(f'diffuse inverse at {t_idx}')
        
        t = (torch.ones(x_t2.size(0)) * t_idx).long().to(self.device)
        if zt is None: zt = torch.randn_like(x_t2, device=self.device)
        sigma_t = torch.sqrt(self.ns.beta(t))

        outter_weight = (1 / torch.sqrt(self.ns.alpha(t)))
        z_weight = (1 - self.ns.alpha(t)) / torch.sqrt(1 - self.ns.alpha_bar(t))

        x_t1 = batch_weight(
            outter_weight, 
            x_t2 - batch_weight(z_weight, self.znet(x_t2, t))
        ) + batch_weight(sigma_t, zt)

        return x_t1

    def sample_interval(self, x_t2: torch.Tensor, t_interval: Tuple[int, int]) -> torch.Tensor: 
        t_end, t_start = t_interval
        assert 0 <= t_start < t_end <= self.T
        for t_idx in (reversed(range(t_start, t_end))): 
            zt = torch.randn_like(x_t2, device=self.device)
            x_t1 = self.sample_step(x_t2, t_idx, zt)
            x_t2 = x_t1
        return x_t2
    
    def full_sample(self, xt: torch.Tensor) -> torch.Tensor: 
        return self.sample_interval(xt, (self.T, 1))

