import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from models.discriminator import Discriminator
from models.generator import Generator
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision.transforms as transforms
from models.utils import get_device


num_epoch = 100
batch_size = 256
latent_dim = 64


if __name__ == "__main__":
    device = get_device()
    data = torchvision.datasets.MNIST(
        root="data/",
        download=True,
        transform=(
            transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        ),
    )

    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    generator = Generator(latent_dim=latent_dim)
    discriminator = Discriminator()

    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=0.0003,
    )
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=0.0003,
    )

    generator.to(device)
    discriminator.to(device)

    loss_fn = nn.BCELoss()  # G的loss
    writer = SummaryWriter()

    for epoch in range(num_epoch):
        g_epoch_loss = 0
        d_epoch_loss = 0
        for index, mini_batch in enumerate(dataloader):
            mini_batch_size = mini_batch[0].size(0)
            generator.train()
            discriminator.train()

            images, _ = mini_batch
            images = images.to(device)

            # --------训练生成器-----------

            # 随机生成一个噪声
            z = torch.normal(0, 1, (mini_batch_size, latent_dim)).to(device)

            # 生成预测
            pred = generator(z)

            g_loss = loss_fn(
                discriminator(pred), torch.ones(mini_batch_size, 1).to(device)
            )
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # --------训练判别器---------

            # 对判别器来说，它有两个任务：
            # 1. 将真实的数据预测为1
            # 2. 错误的数据预测为0

            real_loss = loss_fn(
                discriminator(images), torch.ones(mini_batch_size, 1).to(device)
            )

            fake_loss = loss_fn(
                discriminator(pred.detach()), torch.zeros(mini_batch_size, 1).to(device)
            )  # detach是因为不需要生成器部分的梯度

            d_loss = 0.5 * (real_loss + fake_loss)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            g_epoch_loss += g_loss.item()
            d_epoch_loss += d_loss.item()

            # 保存一下图片
            if index % 1000 == 0:
                writer.add_image("real", torchvision.utils.make_grid(images), epoch)
                writer.add_image("fake", torchvision.utils.make_grid(pred), epoch)
                writer.add_scalar("g_loss", g_loss, epoch * len(dataloader) + index)
                writer.add_scalar("d_loss", d_loss, epoch * len(dataloader) + index)

        print(
            f"Epoch {epoch}/{num_epoch}, G loss: {g_epoch_loss}, D loss: {d_epoch_loss}"
        )

    writer.close()
    check_points_dir = "check_points/"
    if not os.path.exists(check_points_dir):
        os.makedirs(check_points_dir)

    # 保存模型
    torch.save(generator.state_dict(), os.path.join(check_points_dir, "generator.pth"))
    torch.save(
        discriminator.state_dict(), os.path.join(check_points_dir, "discriminator.pth")
    )
