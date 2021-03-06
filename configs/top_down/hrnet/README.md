# Deep high-resolution representation learning for human pose estimation

## Introduction
```
@inproceedings{sun2019deep,
  title={Deep high-resolution representation learning for human pose estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5693--5703},
  year={2019}
}
```

## Results and models

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :----------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py)  | 256x192 | 0.746 | 0.904 | 0.819 | 0.799 | 0.942 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_20200708.log.json) |
| [pose_hrnet_w32](/configs/top_down/hrnet/coco/hrnet_w32_coco_384x288.py)  | 384x288 | 0.760 | 0.906 | 0.829 | 0.810 | 0.943 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288-d9f0d786_20200708.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_384x288_20200708.log.json) |
| [pose_hrnet_w48](/configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py)  | 256x192 | 0.756 | 0.907 | 0.825 | 0.806 | 0.942 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192_20200708.log.json) |
| [pose_hrnet_w48](/configs/top_down/hrnet/coco/hrnet_w48_coco_384x288.py)  | 384x288 | 0.767 | 0.910 | 0.831 | 0.816 | 0.946 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_20200708.log.json) |

### Results on COCO-WholeBody v1.0 val with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | Body AP | Body AR | Foot AP | Foot AR | Face AP | Face AR  | Hand AP | Hand AR | Whole AP | Whole AR | ckpt | log |
| :---- | :--------: | :-----: | :-----: | :-----: | :-----: | :-----: | :------: | :-----: | :-----: | :------: |:-------: |:------: | :------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/coco/hrnet_w32_coco_256x192.py)  | 256x192 | 0.700 | 0.746 | 0.567 | 0.645 | 0.637 | 0.688 | 0.473 | 0.546 | 0.553 | 0.626 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192_20200708.log.json) |


### Results on AIC val set.

| Arch  | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt | log |
| :-------------- | :-----------: | :------: | :------: | :------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/top_down/hrnet/aic/hrnet_w32_aic_256x192.py) | 256x192 | 0.675 | 0.957 | 0.751 | 0.703 | 0.961 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_aic_256x192-30a4e465_20200826.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_aic_256x192_20200826.log.json) |


### Results on MPII val set.

| Arch  | Input Size | Mean | Mean@0.1   | ckpt    | log     |
| :--- | :--------: | :------: | :------: |:------: |:------: |
| [pose_hrnet_w32](/configs/top_down/hourglass/mpii/hrnet_w32_mpii_256x256.py) | 256x256 | 0.900 | 0.379 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256-6c4f923f_20200812.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w32_mpii_256x256_20200812.log.json) |
| [pose_hrnet_w48](/configs/top_down/hourglass/mpii/hrnet_w48_mpii_256x256.py) | 256x256 | 0.900 | 0.383 | [ckpt](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256-92cab7bd_20200812.pth) | [log](https://openmmlab.oss-accelerate.aliyuncs.com/mmpose/top_down/hrnet/hrnet_w48_mpii_256x256_20200812.log.json) |
