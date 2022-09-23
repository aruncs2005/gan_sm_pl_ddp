import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import argparse
from PIL import Image
import time
from torchvision.utils import save_image
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments.lightning_environment import LightningEnvironment
from cleanfid import fid
import os
import glob

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
    

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class GAN(LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = BATCH_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(256, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.log("g_loss", g_loss, prog_bar=True)
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss
        
    def validation_step(self, batch, batch_idx):
        print("inside valiation code")
        imgs, _ = batch
        z = self.validation_z.type_as(imgs)

        # log sampled images'
        sample_imgs = self(z)
        #print(sample_imgs)
        grid = torchvision.utils.make_grid(sample_imgs)
        print("comparing between images of length {} and {}".format(len(imgs),len(sample_imgs)))
        #print(grid)

        gen_images_path = "/tmp/gen_files"
        src_images_path = "/tmp/src_files"
        if self.trainer.is_global_zero:
            if not os.path.exists(gen_images_path):
                os.mkdir(gen_images_path)
            
            if not os.path.exists(src_images_path):
                os.mkdir(src_images_path)
        
        ddp.barrier()
        for img in sample_imgs:
            save_image(img, gen_images_path + "/gen_"+str(time.time()) +".jpeg")
        for img1 in imgs:
            save_image(img1, src_images_path + "/src_"+str(time.time()) +".jpeg")
        
        return {"gen_path":gen_images_path,"src_path":src_images_path}

   
    def validation_step_end(self,batch_parts):
        print("inside validation step end, this should run in one process")
        print("we will run FID Generator here")
        if self.trainer.is_global_zero:
            gen_images_path = batch_parts["gen_path"]
            src_images_path = batch_parts["src_path"]

            score = fid.compute_fid(gen_images_path, src_images_path)

            print("FID Score for the training epoch {} is {}".format(self.epoch, score))
            #delete all files after the FID has finished.
            files = glob.glob(gen_images_path + "/*")
            for f in files:
                os.remove(f)
            sr_files = glob.glob(src_images_path + "/*")
            for f in sr_files:
                os.remove(f)
        ddp.barrier()
        return


        
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []



        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="Saves full model for inference to this dir. Also used if load_full is given to load the model. Note the lack of optimizer state here.",
    )
    
    args = parser.parse_known_args()
    
    env = LightningEnvironment()
    env.world_size = lambda: int(os.environ["WORLD_SIZE"])
    env.global_rank = lambda: int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    num_nodes = int(world_size/num_gpus)
    print("world size:{} Number of GPUS:{} Number of nodes:{}".format(world_size,num_gpus,num_nodes))
    ddp = DDPStrategy(cluster_environment=env, process_group_backend="nccl", accelerator="gpu")
    dm = MNISTDataModule()
    model = GAN(*dm.dims)
    trainer = Trainer(
    devices=num_gpus,
    num_nodes=num_nodes,
    max_epochs=50,
    strategy=ddp,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
    )
    trainer.fit(model, dm)
    