import os
import json
from argparse import ArgumentParser
from PIL import Image
from mmpose.apis import inference_top_down_pose_model, init_pose_model

SCENE_NAMES = [
    '1-HIT_Canteen_frames',
    '2-OCT_Bay_frames',
    '7-Shenzhennorth_Station_frames',
    '3-Xili_Crossroad_frames',
    '4-Nanshan_I_Park_frames',
    '8-Xili_Pedestrian_Street_frames',
    '5-Primary_School_frames',
    '9-Tsinghuasz_Basketball_Court_frames',
    '10-Xinzhongguan_frames',
    '12-Tsinghua_Zhulou_frames',
    '13-Tsinghua_Xicao_frames',
    '11-Tsinghua_ceremony_frames',
    '16-Xili_frames',
    # "14-Dongmen_frames",
    # "15-Huaqiangbei_frames"
]
ROOT_DIR = '/media/luvision/新加卷1/PANDA_each_person_data'


def main():
    """Visualize the demo images.
    Input image edge coordinates as bbox.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()
    assert 'cuda' in args.device

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)

    for scene in SCENE_NAMES:
        print('Processing scene: ', scene)
        scene_root = os.path.join(ROOT_DIR, scene)
        with open(os.path.join(scene_root, scene+'.json'), 'r') as load_f:
            batch_labels = json.load(load_f)
        save_dict = {}
        for pid in batch_labels.keys():
            if batch_labels[pid]:
                print('Processing scene: {} person: {}'.format(scene, pid))
                save_dict[pid] = []
                for batch in batch_labels[pid]:
                    buffer = []
                    images = batch['images']
                    # process each image
                    for img_name in images:
                        img_path = os.path.join(scene_root, pid, img_name)
                        img = Image.open(img_path)
                        width, height = img.size

                        # make person bounding boxes: [x,y,width,height]
                        person_bboxes = [[int(width * 5 / 110), int(height * 5 / 110), int(width * 100 / 110),
                                          int(height * 100 / 110)]]

                        # pose estimate on a single image.
                        pose_results = inference_top_down_pose_model(
                            pose_model, img_path, person_bboxes, format='xywh')
                        buffer.append(pose_results[0]['keypoints'].tolist())
                    save_dict[pid].append(buffer)
        json_string = json.dumps(save_dict, indent=2)
        with open(os.path.join(scene_root, scene+'_skeletons.json'), "w") as f:
            f.write(json_string)
        break


if __name__ == '__main__':
    main()
