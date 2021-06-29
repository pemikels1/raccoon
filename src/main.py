from pathlib import Path
import json
import os
from time import time
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd
import args
from dataset import ObjectDetectionDataSet, park_collate
from train_test import train, test
from horizontal_flip import HorizontalFlip, create_flipped_dataset


def main():

    # parse args.py for the hyperparameters
    global args
    args = args.park_parse_args()

    # set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # get the paths to the folders used
    data_folder = Path(args.data)
    checkpoints_folder = Path(args.checkpoint)
    torch_files_folder = Path(args.torch_files)

    # parse the COCO JSON for the ground truths
    json_path = data_folder / '_annotations.coco.json'
    with open(str(json_path)) as jsonfile:
        data = json.load(jsonfile)

    # create a dataframe for the image data
    img_df = pd.DataFrame(data['images'])

    img_list_ordered = []
    boxes_ordered = []
    labels_ordered = []
    cur_boxes = []
    cur_labels = []

    # loop through the annotations
    for i in range(len(data['annotations']) - 1):

        # id of current annotation
        cur_img_id = data['annotations'][i]['image_id']

        # id of next annotation
        next_img_id = data['annotations'][i + 1]['image_id']

        # append annotatation's bounding box to current target variable
        cur_boxes.append(data['annotations'][i]['bbox'])
        cur_labels.append(1)

        # if ids of current and next annotations are different
        # (could have multiple annotations in same picture)
        if cur_img_id != next_img_id:

            # append path of current image to image list
            cur_img_name = img_df[img_df['id'] == cur_img_id]['file_name'].item()
            img_list_ordered.append(data_folder / cur_img_name)

            # append current target to target list and reset current target
            boxes_ordered.append(cur_boxes)
            labels_ordered.append(cur_labels)

            # reset lists
            cur_boxes = []
            cur_labels = []

    target_file = torch_files_folder / 'target.pt'
    input_file = torch_files_folder / 'input.pt'

    # if no input file already created
    if not os.path.isfile(input_file):

        target_list = []
        cur_target_dict = {}

        # loop through boxes
        for i in range(len(boxes_ordered)):

            # set current label and box
            cur_target_dict['labels'] = labels_ordered[i]
            cur_target_dict['boxes'] = boxes_ordered[i]

            # append to list and reset dictionary
            target_list.append(cur_target_dict)
            cur_target_dict = {}

        # save target file
        torch.save(target_list, target_file)
        torch.save(img_list_ordered, input_file)

    # if input file already created, load it
    else:
        img_list_ordered = torch.load(input_file)

    # create transforms and resize image
    resize = (300, 400)
    transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(resize),
                                    transforms.ToTensor()
    ]
    )

    # create the dataset, converting to xyxy coordinates from xywh coordinates
    dataset = ObjectDetectionDataSet(img_list_ordered, True, target_file, 'xyxy', resize, transform=transform)

    # create a dataset of flipped images
    flipped_dataset = create_flipped_dataset(dataset, labels_ordered, torch_files_folder, resize, transform)

    # concatenate the original dataset and the flipped dataset together to create a new dataset
    concat_dataset = ConcatDataset([dataset, flipped_dataset])

    # calculate lengths for train and test sets
    train_len = round(len(concat_dataset) * args.train_percentage)
    test_len = len(concat_dataset) - train_len

    # randomly split the current dataset into the train and test
    train_dataset, test_dataset = random_split(concat_dataset, [train_len, test_len],
                                               generator=torch.Generator().manual_seed(args.seed))

    # create dataloader for train set
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0,
                                   generator=torch.Generator().manual_seed(args.seed), collate_fn=park_collate)

    # load pretrained model and change last layer to have 2 classes (raccoon or no raccoon)
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # use CUDA for GPU if available, otherwise use CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # switch model to be on device
    model.to(device)

    # initialize optimizer
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)

    best_avg_iou = float('-inf')
    start = time()

    avg_ious = []
    # loop through epochs and train and test the model
    for epoch in range(args.epochs):
        print('epoch:', epoch + 1)
        start = time()
        train(model, epoch + 1, optim, train_data_loader, device)
        end = time()
        print('elapsed time:', end - start)
        start = time()
        test_avg_iou, test_iou_list = test(model, epoch + 1, test_dataset, device, 'test', score_threshold=0.5, nms_thresh=0.1)
        end = time()
        print('elapsed time:', end - start)

        print("Test Avg IOU:", test_avg_iou)

        avg_ious.append(test_avg_iou)

        if test_avg_iou > best_avg_iou:
            print('New checkpoint!')
            torch.save(model.state_dict(), checkpoints_folder / 'model.pt')
            best_avg_iou = test_avg_iou

    end = time()
    print("ELAPSED TIME:", end - start)


if __name__ == '__main__':
    main()
