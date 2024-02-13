import torch.nn as nn
import torch
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    # Source for the VectorQuantizerEMA module: https://github.com/zalandoresearch/pytorch-vq-vae
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def get_quantized(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        e_quantized_loss = F.mse_loss(quantized,inputs.detach())
        loss = self._commitment_cost * e_latent_loss + e_quantized_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    # Source for the VectorQuantizerEMA module: https://github.com/zalandoresearch/pytorch-vq-vae
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def get_quantized(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings



class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, groups=1):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False, groups=groups),
            nn.ReLU(False),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False, groups=groups)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, groups=1):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens, groups=groups)
                                      for _ in range(self._num_residual_layers)])
        self.relu = nn.ReLU(False)

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return self.relu(x)


class EncoderBot(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderBot, self).__init__()
        self.input_conv_depth =nn.Conv2d(in_channels=1,
                                 out_channels=num_hiddens//4,
                                 kernel_size=1,
                                 stride=1, padding=0)
        self.input_conv_rgb =nn.Conv2d(in_channels=in_channels-1,
                                 out_channels=num_hiddens//4,
                                 kernel_size=1,
                                 stride=1, padding=0)

        self._conv_1 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1, groups=2)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1, groups=2)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1, groups=2)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens, groups=2)

        self.relu = nn.ReLU(False)

    def forward(self, inputs):
        x_d = self.input_conv_depth(inputs[:,:1,:,:])
        x_rgb = self.input_conv_rgb(inputs[:,1:,:,:])
        x = torch.cat((x_d,x_rgb),dim=1)
        x = self.relu(x)
        x = self._conv_1(x)
        x = self.relu(x)

        x = self._conv_2(x)
        x = self.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class EncoderTop(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderTop, self).__init__()
        self._conv_1 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1,groups=2)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1,groups=2)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens, groups=2)

        self.relu = nn.ReLU(False)


    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self.relu(x)

        x = self._conv_2(x)
        x = self.relu(x)

        x = self._residual_stack(x)
        return x


class DecoderBot(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=3):
        super(DecoderBot, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1, groups=2)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens, groups=2)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1, groups=2)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=num_hiddens//4,
                                                kernel_size=4,
                                                stride=2, padding=1, groups=2)
        self._conv_out_depth = nn.Conv2d(in_channels=num_hiddens//8,
                                 out_channels=1,
                                 kernel_size=1,
                                 stride=1, padding=0, groups=1)
        self._conv_out_rgb = nn.Conv2d(in_channels=num_hiddens//8,
                                 out_channels=3,
                                 kernel_size=1,
                                 stride=1, padding=0, groups=1)

        self.relu = nn.ReLU(False)


    def forward(self, inputs):
        b,c,h,w = inputs.shape
        in_t = inputs[:,:c//2,:,:]
        in_b = inputs[:,c//2:,:,:]

        b,c,h,w = in_t.shape
        in_t_d = in_t[:,:c//2,:,:]
        in_t_rgb = in_t[:,c//2:,:,:]
        in_b_d = in_b[:,:c//2,:,:]
        in_b_rgb = in_b[:,c//2:,:,:]
        in_joined = torch.cat((in_t_d,in_b_d, in_t_rgb,in_b_rgb),dim=1)
        x = self._conv_1(in_joined)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = self.relu(x)
        x = self._conv_trans_2(x)
        c = x.shape[1]
        x_d = x[:,:c//2,:,:]
        x_rgb = x[:,c//2:,:,:]
        out_d = self._conv_out_depth(x_d)
        out_rgb = self._conv_out_rgb(x_rgb)
        x_out = torch.cat((out_d,out_rgb),dim=1)

        return x_out

class PreVQBot(nn.Module):
    def __init__(self, num_hiddens, embedding_dim):
        super(PreVQBot, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding_dim = embedding_dim
        self._pre_vq_conv_bot = nn.Conv2d(in_channels=num_hiddens + embedding_dim,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1,groups=2)



    def forward(self, inputs):

        #feat = torch.cat((enc_b, up_quantized_t), dim=1)
        in_enc_b = inputs[:,:self.num_hiddens,:,:]
        in_up_t = inputs[:,self.num_hiddens:,:,:]

        in_enc_b_d = in_enc_b[:,:self.num_hiddens//2,:,:]
        in_enc_b_rgb = in_enc_b[:,self.num_hiddens//2:,:,:]
        in_up_t_d = in_up_t[:,:self.embedding_dim//2,:,:]
        in_up_t_rgb = in_up_t[:,self.embedding_dim//2:,:,:]

        in_joined = torch.cat((in_enc_b_d,in_up_t_d, in_enc_b_rgb,in_up_t_rgb),dim=1)
        x = self._pre_vq_conv_bot(in_joined)

        return x

class DiscreteLatentModelGroups(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim,
                 commitment_cost, decay=0, test=False, in_channels=3, out_channels=3):
        super(DiscreteLatentModelGroups, self).__init__()
        self.test = test
        self._encoder_t = EncoderTop(num_hiddens, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)

        self._encoder_b = EncoderBot(in_channels, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)


        self._pre_vq_conv_top = nn.Conv2d(in_channels=num_hiddens,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1,groups=2)

        self._pre_vq_conv_bot = PreVQBot(num_hiddens,embedding_dim)

        self._vq_vae_top = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        self._vq_vae_bot = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        self._decoder_b = DecoderBot(embedding_dim*2,
                                     num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens, out_channels=out_channels)


        self.upsample_t = nn.ConvTranspose2d(
            embedding_dim, embedding_dim, 4, stride=2, padding=1, groups=2
        )


    def forward(self, x):
        #Encoder Hi
        enc_b = self._encoder_b(x)

        #Encoder Lo -- F_Lo
        enc_t = self._encoder_t(enc_b)
        zt = self._pre_vq_conv_top(enc_t)

        # Quantize F_Lo with K_Lo
        loss_t, quantized_t, perplexity_t, encodings_t = self._vq_vae_top(zt)
        # Upsample Q_Lo
        up_quantized_t = self.upsample_t(quantized_t)

        # Concatenate and transform the output of Encoder_Hi and upsampled Q_lo -- F_Hi
        feat = torch.cat((enc_b, up_quantized_t), dim=1)
        zb = self._pre_vq_conv_bot(feat)

        # Quantize F_Hi with K_Hi
        loss_b, quantized_b, perplexity_b, encodings_b = self._vq_vae_bot(zb)

        # Concatenate Q_Hi and Q_Lo and input it into the General appearance decoder
        quant_join = torch.cat((up_quantized_t, quantized_b), dim=1)
        recon_fin = self._decoder_b(quant_join)

        #return loss_b, loss_t, recon_fin, encodings_t, encodings_b, quantized_t, quantized_b
        return loss_b, loss_t, recon_fin, quantized_t, quantized_b


