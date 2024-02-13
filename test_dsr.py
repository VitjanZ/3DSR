import torch
import torch.nn.functional as F
from data_loader import MVTecDRAEMTestDataset
from torch.utils.data import DataLoader
import os
import numpy as np
from dsr_model import SubspaceRestrictionModule, ImageReconstructionNetwork, AnomalyDetectionModule
from discrete_model_groups import DiscreteLatentModelGroups
from sklearn.metrics import roc_auc_score, average_precision_score
from au_pro_util import calculate_au_pro

def test(obj_names, mvtec_path, out_path, run_name_base):
    total_ap_pixel = []
    total_auroc_pixel = []
    total_ap = []
    total_auroc = []
    total_aupro = []

    dada_model_path = "./checkpoints/DADA_RGB_D.pckl"

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
    model.load_state_dict(torch.load(dada_model_path, map_location='cuda:0'))
    model.eval()

    embedder_hi = model._vq_vae_bot
    embedder_lo = model._vq_vae_top



    for obj_name in obj_names:
        img_dim = 384
        run_name = run_name_base+obj_name+'_'


        sub_res_model_lo = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_model_lo.load_state_dict(torch.load(out_path+run_name+"_recon_lo.pckl", map_location='cuda:0'))
        sub_res_model_hi = SubspaceRestrictionModule(embedding_size=embedding_dim)
        sub_res_model_hi.load_state_dict(torch.load(out_path+run_name+"_recon_hi.pckl", map_location='cuda:0'))
        sub_res_model_lo.cuda()
        sub_res_model_lo.eval()
        sub_res_model_hi.cuda()
        sub_res_model_hi.eval()

        # Define the anomaly detection module - UNet-based network
        decoder_seg = AnomalyDetectionModule(in_channels=8, base_width=32)
        decoder_seg.load_state_dict(torch.load(out_path+run_name+"_seg.pckl", map_location='cuda:0'))
        decoder_seg.cuda()
        decoder_seg.eval()

        # Image reconstruction network reconstructs the image from discrete features.
        # It is trained for a specific object
        model_decode = ImageReconstructionNetwork(embedding_dim * 2,
                   num_hiddens,
                   num_residual_layers,
                   num_residual_hiddens, out_channels=4)
        model_decode.load_state_dict(torch.load(out_path+run_name+"_decode.pckl", map_location='cuda:0'))
        model_decode.cuda()
        model_decode.eval()

        dataset = MVTecDRAEMTestDataset(mvtec_path + obj_name + "/test/*/xyz/",
                                        resize_shape=[img_dim, img_dim])
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, num_workers=0)

        #img_dim=224
        total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
        total_pixel_scores_2d = np.zeros((len(dataset),img_dim, img_dim))
        total_gt_pixel_scores_2d = np.zeros((len(dataset),img_dim, img_dim))

        mask_cnt = 0

        total_gt = []
        total_score = []

        for i_batch, sample_batched in enumerate(dataloader):

            depth_image = sample_batched["image"].cuda()

            is_normal = sample_batched["has_anomaly"].detach().numpy()[0, 0]
            total_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))

            rgb_image = sample_batched["rgb_image"].cuda()

            in_image = torch.cat((depth_image, rgb_image), dim=1)
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

            out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[:, 1:, :, :], 11, stride=1,
                                                               padding=11 // 2).cpu().detach().numpy()
            image_score = np.max(out_mask_averaged)
            flat_out_mask = out_mask_averaged[0,0,:,:].flatten()

            total_score.append(image_score)

            flat_true_mask = true_mask_cv.flatten()
            total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            total_pixel_scores_2d[mask_cnt] = out_mask_averaged[0,0,:,:]
            total_gt_pixel_scores_2d[mask_cnt] = true_mask_cv[:,:,0]
            mask_cnt += 1

        total_score = np.array(total_score)
        total_gt = np.array(total_gt)
        auroc = roc_auc_score(total_gt, total_score)
        ap = average_precision_score(total_gt, total_score)

        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
        aupro, _ = calculate_au_pro([total_gt_pixel_scores_2d[x] for x in range(total_gt_pixel_scores_2d.shape[0])], [total_pixel_scores_2d[x] for x in range(total_pixel_scores_2d.shape[0])])

        print("------------------")
        print(obj_name)
        print("AUC Image: " + str(auroc))
        print("AP Image: " + str(ap))
        print("AUC Pixel: " + str(auroc_pixel))
        print("AP Pixel: " + str(ap_pixel))
        print("AUPRO: " + str(aupro))

        total_aupro.append(aupro)
        total_auroc_pixel.append(auroc_pixel)
        total_auroc.append(auroc)
        total_ap.append(ap)
        total_ap_pixel.append(ap_pixel)
    print("--------MEAN---------------------------------------")
    print("AUC Image: " + str(np.mean(total_auroc)))
    print("AP Image: " + str(np.mean(total_ap)))
    print("AUC Pixel: " + str(np.mean(total_auroc_pixel)))
    print("AP Pixel: " + str(np.mean(total_ap_pixel)))
    print("AUPRO: " + str(np.mean(total_aupro)))

    print("AUC",*[np.round(x*100,2) for x in total_auroc],np.round(np.mean(total_auroc)*100,2))
    print("AUCp",*[np.round(x*100,2) for x in total_auroc_pixel],np.round(np.mean(total_auroc_pixel)*100,2))
    print("AUPRO",*[np.round(x*100,2) for x in total_aupro],np.round(np.mean(total_aupro)*100,2))
    print("AP",*[np.round(x*100,2) for x in total_ap],np.round(np.mean(total_ap)*100,2))

if __name__=="__main__":
    obj_names = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--out_path', action='store', type=str, required=True)
    parser.add_argument('--run_name', action='store', type=str, required=True)

    args = parser.parse_args()
    with torch.cuda.device(args.gpu_id):
        test(obj_names,args.data_path, args.out_path, args.run_name)

