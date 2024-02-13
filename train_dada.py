import torch
import torch.nn.functional as F
from data_loader_vq_vae import DADADataset
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
import sys
from discrete_model_groups import DiscreteLatentModelGroups

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(imagenet_path):
    lr = 2e-4
    epochs = 100 # 100k
    batch_size = 64
    run_name = 'DADA_l1_'

    num_hiddens = 256
    num_residual_hiddens = 128
    num_residual_layers = 2
    embedding_dim = 256
    num_embeddings = 2048
    commitment_cost = 0.25
    decay = 0.99

    model = DiscreteLatentModelGroups(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                                embedding_dim,
                                commitment_cost, decay, in_channels=4, out_channels=4)

    model.cuda()
    model.apply(weights_init)
    model.train()

    optimizer = torch.optim.Adam([
                                  {"params": model.parameters(), "lr": lr}])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[epochs*0.8, epochs*0.9],gamma=0.1, last_epoch=-1)


    dataset = DADADataset(imagenet_path, resize_shape=[256, 256], bs=batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    n_train = len(dataset)

    l2_rgb_loss_acc = 0
    l2_total_loss_acc = 0
    l2_d_loss_acc = 0

    n_iter = 0
    for epoch in range(epochs):
        for i_batch, sample_batched in enumerate(dataloader):
            depth_image = sample_batched["image"].cuda()
            rgb_image = sample_batched["rgb_image"].cuda()

            optimizer.zero_grad()

            model_in = torch.cat((depth_image, rgb_image), dim=1).float()

            loss_b, loss_t, recon_out, _, _ = model(model_in)
            loss_vq = loss_b + loss_t

            recon_depth = recon_out[:,:1,:,:]
            recon_rgb = recon_out[:,1:,:,:]

            # Using L1 loss may work better and lead to improved reconstructions
            #l2_recon_d_loss = torch.mean((depth_image - recon_depth)**2)
            l2_recon_d_loss = torch.mean(torch.abs(depth_image - recon_depth))
            #l2_recon_rgb_loss = torch.mean((rgb_image - recon_rgb)**2)
            l2_recon_rgb_loss = torch.mean(torch.abs(rgb_image - recon_rgb))
            #l2_recon_loss = torch.mean((model_in - recon_out)**2)
            l2_recon_loss = torch.mean(torch.abs(model_in - recon_out))
            recon_loss = l2_recon_loss + loss_vq
            loss = recon_loss

            loss.backward()
            optimizer.step()


            l2_rgb_loss_acc += l2_recon_rgb_loss.item()
            l2_total_loss_acc += loss.item()
            l2_d_loss_acc += l2_recon_d_loss.item()
            if n_iter % 100 == 0:
                l2_rgb_loss_acc = l2_rgb_loss_acc / 100.0
                l2_d_loss_acc = l2_d_loss_acc / 100.0
                l2_total_loss_acc = l2_total_loss_acc / 100.0
                print("Epoch", epoch, " RGB Loss: ", l2_rgb_loss_acc, " Depth Loss: ", l2_d_loss_acc, " Total Loss: ", l2_total_loss_acc)
                l2_rgb_loss_acc = 0
                l2_d_loss_acc = 0
                l2_total_loss_acc = 0

            n_iter +=1

        scheduler.step()
        if not os.path.exists('./checkpoints'):
            os.mkdir('checkpoints')
        torch.save(model.state_dict(), "./checkpoints/"+run_name+".pckl")


if __name__=="__main__":
    with torch.cuda.device(int(sys.argv[1])):
        train_on_device(sys.argv[2])





