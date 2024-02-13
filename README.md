# 3DSR
Official implementation of the WACV2024 paper: Cheating Depth: Enhancing 3D Surface Anomaly Detection via Depth Simulation

## Usage
### Checkpoints
The pretrained VQ models for depth (DADA_D.pckl) and RGB+depth (DADA_RGB_D.pckl) are in the checkpoints folder. Training the model on depth can be done with train_dada_depth.py. Training the VQ model on RGB+depth can be done with train_dada.py, however the proper paths must be set for ImageNet or any other RGB dataset. (see usage in train_dada_sbatch.sh)

## Tran 3DSR
For training you can use the train_dsr_depth.py and train_dsr.py files for training in the depth-only and the RGB+depth setup, respectively. You can also use the sbatch scripts as an example if you have access to a slurm cluster. By default the models in the checkpoints folder are used as the VQ model for training 3DSR.
### Train 3DSR depth
```
python train_dsr_depth.py --gpu_id 0 --obj_id $i --lr 0.0002 --bs 16 --epochs 120 --data_path $MVTEC3D_PATH --out_path $OUT_PATH --run_name $RUN_NAME
```
### Train 3DSR RGB + depth
```
python train_dsr.py --gpu_id 0 --obj_id $i --lr 0.0002 --bs 16 --epochs 120 --data_path $MVTEC3D_PATH --out_path $OUT_PATH --run_name $RUN_NAME
```

### Evaluate
```
python test_dsr.py --gpu_id 0 --data_path $MVTEC3D_PATH --out_path $OUT_PATH --run_name $RUN_NAME 
```


### Pretrained checkpoints
Will be uploaded soon.

### Reference
```
@inproceedings{zavrtanik2024cheating,
  title={Cheating Depth: Enhancing 3D Surface Anomaly Detection via Depth Simulation},
  author={Zavrtanik, Vitjan and Kristan, Matej and Sko{\v{c}}aj, Danijel},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2164--2172},
  year={2024}
}
```
