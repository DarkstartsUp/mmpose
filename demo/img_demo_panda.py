import os
import json
from argparse import ArgumentParser
from PIL import Image
from mmpose.apis import inference_pose_model, init_pose_model, vis_pose_result


def main():
    """Visualize the demo images.
    Input image edge coordinates as bbox. 
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for detection')
    parser.add_argument('pose_checkpoint', help='Checkpoint file')
    parser.add_argument('--img-root', type=str, default='', help='Image root')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='Root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert 'cuda' in args.device

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device)

    img_name_list = []
    file_list = os.listdir(args.img_root)
    for file_name in sorted(file_list):
        if '.jpg' in file_name:
            img_name_list.append(file_name)
    save_list = []
    # process each image
    for i, img_name in enumerate(img_name_list):
        img_path = os.path.join(args.img_root, img_name)
        img = Image.open(img_path)
        width, height = img.size

        # make person bounding boxes: [x,y,width,height]
        person_bboxes = []
        person_bboxes.append([int(width * 5 / 110), int(height * 5 / 110), int(width * 100 / 110), int(height * 100 / 110)])

        # test a single image, with a list of bboxes.
        pose_results = inference_pose_model(
            pose_model, img_path, person_bboxes, format='xywh')
        print(len(pose_results[0]['keypoints'].tolist()))
        save_list.append(pose_results[0]['keypoints'].tolist())

        if args.out_img_root == '':
            out_file = None
        else:
            out_file = os.path.join(args.out_img_root, f'vis_{i}.jpg')

        # show the results
        # vis_pose_result(
        #     pose_model,
        #     img_path,
        #     pose_results,
        #     skeleton=skeleton,
        #     kpt_score_thr=args.kpt_thr,
        #     show=args.show,
        #     out_file=out_file)
    json_string = json.dumps(save_list, indent=2)
    with open(os.path.join(args.out_img_root, 'results.json'), "w") as f:
        f.write(json_string)


if __name__ == '__main__':
    main()
