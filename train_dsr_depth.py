import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset, MVTecDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
import os
import numpy as np
from dsr_model import SubspaceRestrictionModule, ImageReconstructionNetwork, AnomalyDetectionModule
from discrete_model import DiscreteLatentModel
from loss import FocalLoss
from sklearn.metrics import roc_auc_score


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def shuffle_patches(x, patch_size):
    # divide the batch of images into non-overlapping patches
    u = torch.nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = torch.nn.functional.fold(pu, x.shape[-2:], kernel_size=patch_size, stride=patch_size, padding=0)
    return f

def generate_fake_anomalies_joined(features,embeddings, memory_torch_original, mask, strength=None):
    random_embeddings = torch.zeros((embeddings.shape[0],embeddings.shape[2]*embeddings.shape[3], memory_torch_original.shape[1]))
    inputs = features.permute(0, 2, 3, 1).contiguous()

    for k in range(embeddings.shape[0]):
        memory_torch = memory_torch_original
        flat_input = inputs[k].view(-1, memory_torch.shape[1])

        distances_b = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(memory_torch ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, memory_torch.t()))

        percentage_vectors = strength[k]
        topk = max(1, min(int(percentage_vectors * memory_torch.shape[0]) + 1, memory_torch.shape[0] - 1))
        values, topk_indices = torch.topk(distances_b, topk, dim=1, largest=False)
        topk_indices = topk_indices[:, int(memory_torch.shape[0] * 0.05):]
        topk = topk_indices.shape[1]

        random_indices_hik = torch.randint(topk, size=(topk_indices.shape[0],))
        random_indices_t = topk_indices[torch.arange(random_indices_hik.shape[0]),random_indices_hik]
        random_embeddings[k] = memory_torch[random_indices_t,:]
    random_embeddings = random_embeddings.reshape((random_embeddings.shape[0],embeddings.shape[2],embeddings.shape[3],random_embeddings.shape[2]))
    random_embeddings_tensor = random_embeddings.permute(0,3,1,2).cuda()

    use_shuffle = torch.rand(1)[0].item()
    if use_shuffle > 0.5:
        psize_factor = torch.randint(0, 4, (1,)).item() # 1, 2, 4, 8
        random_embeddings_tensor = shuffle_patches(embeddings,2**psize_factor)


    down_ratio_y = int(mask.shape[2]/embeddings.shape[2])
    down_ratio_x = int(mask.shape[3]/embeddings.shape[3])
    anomaly_mask = torch.nn.functional.max_pool2d(mask, (down_ratio_y, down_ratio_x)).float()


    anomaly_embedding = anomaly_mask * random_embeddings_tensor + (1.0 - anomaly_mask) * embeddings

    return anomaly_embedding

def evaluate_model(model, sub_res_model_lo, sub_res_model_hi, embedder_hi, embedder_lo, model_decode, decoder_seg, visualizer, obj_name, n_iter, img_min, img_max, mvtec_path):
    img_dim = 384
    dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/*/xyz/", resize_shape=[img_dim,img_dim], img_min=img_min, img_max=img_max)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    total_gt = []
    total_score = []
    loss_recon = 0.0
    cnt_normal = 0

    mask_cnt = 0


    for i_batch, sample_batched in enumerate(dataloader):

        depth_image = sample_batched["image"].cuda()
        true_mask = sample_batched["mask"].cuda()
        true_mask_cv = true_mask.cpu().detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        total_gt.append(1.0 if np.sum(true_mask_cv) > 0 else 0.0)

        in_image = depth_image
        _, _, recon_out, embeddings_lo, embeddings_hi = model(in_image)
        recon_image_general = recon_out

        _, recon_embeddings_hi, _ = sub_res_model_hi(embeddings_hi, embedder_hi)
        _, recon_embeddings_lo, _ = sub_res_model_lo(embeddings_lo, embedder_lo)

        # Reconstruct the image from the reconstructed features
        # with the object-specific image reconstruction module
        up_quantized_recon_t = model.upsample_t(recon_embeddings_lo)
        quant_join = torch.cat((up_quantized_recon_t, recon_embeddings_hi), dim=1)
        recon_image_recon = model_decode(quant_join)

        # Generate the anomaly segmentation map
        out_mask = decoder_seg(recon_image_recon.detach(), recon_image_general.detach())
        out_mask_sm = torch.softmax(out_mask, dim=1)

        out_mask_cv = out_mask_sm[0,1,:,:].detach().cpu().numpy()
        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:,1:,:,:], 11, stride=1,
                                                           padding=11 // 2).cpu().detach().numpy()
        image_score = np.max(out_mask_averaged)
        total_score.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1


    total_score = np.array(total_score)
    total_gt = np.array(total_gt)
    auroc = roc_auc_score(total_gt, total_score)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    print(str(n_iter)," ",obj_name," AUC IM: ",str(auroc)," AUC Pix: ",str(auroc_pixel))
    return auroc

def train_on_device(obj_names, mvtec_path, out_path, lr, batch_size, epochs, run_name_base):
    dada_model_path = "./checkpoints/DADA_D.pckl"

    img_dim = 384
    num_hiddens = 256
    num_residual_hiddens = 128
    num_residual_layers = 2
    embedding_dim = 256
    num_embeddings = 2048
    commitment_cost = 0.25
    decay = 0.99

    model = DiscreteLatentModel(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings,
                                embedding_dim,
                                commitment_cost, decay, in_channels=1, out_channels=1)
    model.cuda()
    model.load_state_dict(torch.load(dada_model_path, map_location='cuda:0'))
    model.eval()

    # Modules using the codebooks K_hi and K_lo for feature quantization
    embedder_hi = model._vq_vae_bot
    embedder_lo = model._vq_vae_top

    for obj_name in obj_names:
        run_name = run_name_base+"_" + str(lr) + '_' + str(epochs) + '_bs' + str(batch_size) + "_" + obj_name + '_'
        visualizer=None

        # Define the subspace restriction modules - Encoder decoder networks
        sub_res_model_lo = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_model_hi = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_model_lo.cuda()
        sub_res_model_hi.cuda()

        # Define the anomaly detection module - UNet-based network
        #decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=64)
        #decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=128)
        decoder_seg = AnomalyDetectionModule(in_channels=2, base_width=32)
        decoder_seg.cuda()
        decoder_seg.apply(weights_init)


        # Image reconstruction network reconstructs the image from discrete features.
        # It is trained for a specific object
        model_decode = ImageReconstructionNetwork(embedding_dim * 2,
                   num_hiddens,
                   num_residual_layers,
                   num_residual_hiddens, out_channels=1)
        model_decode.cuda()
        model_decode.apply(weights_init)



        optimizer = torch.optim.Adam([
                                      {"params": sub_res_model_lo.parameters(), "lr": lr},
                                      {"params": sub_res_model_hi.parameters(), "lr": lr},
                                      {"params": model_decode.parameters(), "lr": lr},
                                      {"params": decoder_seg.parameters(), "lr": lr}])

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[int(epochs*0.80)],gamma=0.1, last_epoch=-1)

        loss_focal = FocalLoss()
        dataset = MVTecDRAEMTrainDataset(mvtec_path + obj_name  + "/train/good/", resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=2)


        n_iter = 0.0
        for epoch in range(epochs):
            for i_batch, sample_batched in enumerate(dataloader):

                depth_image = sample_batched["image"].cuda()
                anomaly_mask = sample_batched["mask"].cuda()

                optimizer.zero_grad()

                with torch.no_grad():
                    in_image = depth_image

                    anomaly_strength_lo = (torch.rand(in_image.shape[0]) * 0.90 + 0.1).cuda()
                    anomaly_strength_hi = (torch.rand(in_image.shape[0]) * 0.90 + 0.1).cuda()
                    # Extract features from the discrete model
                    enc_b = model._encoder_b(in_image)
                    enc_t = model._encoder_t(enc_b)
                    zt = model._pre_vq_conv_top(enc_t)

                    # Quantize the extracted features
                    _, quantized_t, _, _ = embedder_lo(zt)

                    # Generate feature-based anomalies on F_lo
                    anomaly_embedding_lo = generate_fake_anomalies_joined(zt, quantized_t,
                                                                           embedder_lo._embedding.weight,
                                                                           anomaly_mask, strength=anomaly_strength_lo)

                    # Upsample the extracted quantized features and the quantized features augmented with anomalies
                    up_quantized_t = model.upsample_t(anomaly_embedding_lo)
                    up_quantized_t_real = model.upsample_t(quantized_t)
                    feat = torch.cat((enc_b, up_quantized_t), dim=1)
                    feat_real = torch.cat((enc_b, up_quantized_t_real), dim=1)
                    zb = model._pre_vq_conv_bot(feat)
                    zb_real = model._pre_vq_conv_bot(feat_real)
                    # Quantize the upsampled features - F_hi
                    _, quantized_b, _, _ = embedder_hi(zb)
                    _, quantized_b_real, _, _ = embedder_hi(zb_real)

                    # Generate feature-based anomalies on F_hi
                    anomaly_embedding = generate_fake_anomalies_joined(zb, quantized_b,
                                                                          embedder_hi._embedding.weight, anomaly_mask
                                                                         , strength=anomaly_strength_hi)

                    use_both = torch.randint(0, 2,(in_image.shape[0],1,1,1)).cuda().float()
                    use_lo = torch.randint(0, 2,(in_image.shape[0],1,1,1)).cuda().float()
                    use_hi = (1 - use_lo)
                    anomaly_embedding_hi_usebot = generate_fake_anomalies_joined(zb_real,
                                                                         quantized_b_real,
                                                                         embedder_hi._embedding.weight,
                                                                         anomaly_mask, strength=anomaly_strength_hi)
                    anomaly_embedding_lo_usebot = quantized_t
                    anomaly_embedding_hi_usetop = quantized_b_real
                    anomaly_embedding_lo_usetop = anomaly_embedding_lo
                    anomaly_embedding_hi_not_both =  use_hi * anomaly_embedding_hi_usebot + use_lo * anomaly_embedding_hi_usetop
                    anomaly_embedding_lo_not_both =  use_hi * anomaly_embedding_lo_usebot + use_lo * anomaly_embedding_lo_usetop
                    anomaly_embedding_hi = (anomaly_embedding * use_both + anomaly_embedding_hi_not_both * (1.0 - use_both)).detach().clone()
                    anomaly_embedding_lo = (anomaly_embedding_lo * use_both + anomaly_embedding_lo_not_both * (1.0 - use_both)).detach().clone()

                    anomaly_embedding_hi_copy = anomaly_embedding_hi.clone()
                    anomaly_embedding_lo_copy = anomaly_embedding_lo.clone()

                # Restore the features to normality with the Subspace restriction modules
                recon_feat_hi, recon_embeddings_hi, loss_b = sub_res_model_hi(anomaly_embedding_hi_copy, embedder_hi)
                recon_feat_lo, recon_embeddings_lo, loss_b_lo = sub_res_model_lo(anomaly_embedding_lo_copy, embedder_lo)

                # Reconstruct the image from the anomalous features with the general appearance decoder
                up_quantized_anomaly_t = model.upsample_t(anomaly_embedding_lo)
                quant_join_anomaly = torch.cat((up_quantized_anomaly_t, anomaly_embedding_hi), dim=1)
                recon_image_general = model._decoder_b(quant_join_anomaly)


                # Reconstruct the image from the reconstructed features
                # with the object-specific image reconstruction module
                up_quantized_recon_t = model.upsample_t(recon_embeddings_lo)
                quant_join = torch.cat((up_quantized_recon_t, recon_embeddings_hi), dim=1)
                recon_image_recon = model_decode(quant_join)

                # Generate the anomaly segmentation map
                #out_mask = decoder_seg(recon_image_recon.detach(),recon_image_general.detach())

                out_mask = decoder_seg(recon_image_recon,recon_image_general)
                #out_mask = decoder_seg(recon_image_recon,recon_image_general)
                out_mask_sm = torch.softmax(out_mask, dim=1)

                # Calculate losses
                loss_feat_hi = torch.nn.functional.mse_loss(recon_feat_hi, quantized_b_real.detach())
                loss_feat_lo = torch.nn.functional.mse_loss(recon_feat_lo, quantized_t.detach())
                loss_l2_recon_img = torch.nn.functional.mse_loss(in_image, recon_image_recon)
                total_recon_loss = loss_feat_lo + loss_feat_hi + loss_l2_recon_img*10

                # Resize the ground truth anomaly map to closely match the augmented features
                down_ratio_x_hi = int(anomaly_mask.shape[3] / quantized_b.shape[3])
                anomaly_mask_hi = torch.nn.functional.max_pool2d(anomaly_mask,
                                                                  (down_ratio_x_hi, down_ratio_x_hi)).float()
                anomaly_mask_hi = torch.nn.functional.interpolate(anomaly_mask_hi, scale_factor=down_ratio_x_hi)
                down_ratio_x_lo = int(anomaly_mask.shape[3] / quantized_t.shape[3])
                anomaly_mask_lo = torch.nn.functional.max_pool2d(anomaly_mask,
                                                                  (down_ratio_x_lo, down_ratio_x_lo)).float()
                anomaly_mask_lo = torch.nn.functional.interpolate(anomaly_mask_lo, scale_factor=down_ratio_x_lo)
                anomaly_mask = anomaly_mask_lo * use_both + (
                            anomaly_mask_lo * use_lo + anomaly_mask_hi * use_hi) * (1.0 - use_both)

                #anomaly_mask = anomaly_mask * anomaly_type_sum
                # Calculate the segmentation loss
                segment_loss = loss_focal(out_mask_sm, anomaly_mask)
                l1_mask_loss = torch.mean(torch.abs(out_mask_sm - torch.cat((1.0 - anomaly_mask, anomaly_mask), dim=1)))
                segment_loss = segment_loss + l1_mask_loss

                loss = segment_loss + total_recon_loss
                loss.backward()
                optimizer.step()

                n_iter +=1

            scheduler.step()


            if epoch % 2 == 0:
                with torch.no_grad():
                    evaluate_model(model, sub_res_model_lo, sub_res_model_hi, embedder_hi, embedder_lo, model_decode,
                                   decoder_seg, visualizer, obj_name, n_iter, dataset.global_min, dataset.global_max, mvtec_path)

            if (epoch+1) % 5 == 0:
                # Save models
                if not os.path.exists(out_path+"checkpoints/"):
                    os.makedirs(out_path+"checkpoints/")
                torch.save(decoder_seg.state_dict(), out_path+"checkpoints/"+run_name+"_seg.pckl")
                torch.save(sub_res_model_lo.state_dict(), out_path+"checkpoints/"+run_name+"_recon_lo.pckl")
                torch.save(sub_res_model_hi.state_dict(), out_path+"checkpoints/"+run_name+"_recon_hi.pckl")
                torch.save(model_decode.state_dict(), out_path+"checkpoints/"+run_name+"_decode.pckl")


        with torch.no_grad():
            print(run_name)
            evaluate_model(model, sub_res_model_lo, sub_res_model_hi, embedder_hi, embedder_lo, model_decode,
                           decoder_seg, visualizer, obj_name, n_iter, dataset.global_min, dataset.global_max, mvtec_path)

        if not os.path.exists(out_path + "checkpoints/"):
            os.makedirs(out_path + "checkpoints/")
        torch.save(decoder_seg.state_dict(), out_path + "checkpoints/" + run_name + "_seg.pckl")
        torch.save(sub_res_model_lo.state_dict(), out_path + "checkpoints/" + run_name + "_recon_lo.pckl")
        torch.save(sub_res_model_hi.state_dict(), out_path + "checkpoints/" + run_name + "_recon_hi.pckl")
        torch.save(model_decode.state_dict(), out_path + "checkpoints/" + run_name + "_decode.pckl")

    return model, sub_res_model_hi, sub_res_model_lo, model_decode, decoder_seg

if __name__=="__main__":
    obj_classes = [["cable_gland"], ["bagel"], ["cookie"], ["carrot"], ["dowel"], ["foam"], ["peach"], ["potato"],
                   ["tire"], ["rope"]]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--out_path', action='store', type=str, required=True)
    parser.add_argument('--run_name', action='store', type=str, required=True)

    args = parser.parse_args()


    with torch.cuda.device(args.gpu_id):
        train_on_device(obj_classes[int(args.obj_id)],args.data_path, args.out_path, args.lr, args.bs, args.epochs, args.run_name)
