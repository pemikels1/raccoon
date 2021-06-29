import torch
import numpy as np
import os
from dataset import ObjectDetectionDataSet


# flip image and bounding boxes
class HorizontalFlip(object):

    # initialize variables
    def __init__(self, img, bboxes):
        self.img = img
        self.bboxes = bboxes['boxes']

    def __call__(self):

        self.img = self.img.permute(1, 2, 0)

        # get the center of the image (the symmetrical line to flip), and repeat it (appearing once for each point)
        img_center = torch.tensor(self.img.size()[:2]).div(2)
        img_center = img_center[1].repeat(2).to(torch.int64)

        # calculate the difference from the center for each point's x coordinate and double it
        diff_from_center = img_center - self.bboxes[:, [0, 2]]
        double_diff_from_center = 2 * diff_from_center

        # add each difference to the respective point's x coordinate
        self.bboxes[:, [0, 2]] += double_diff_from_center

        # when flipping horizontally, top left coordinate will be in top right and bottom right coordinate will be in
        # bottom left; need to convert to top left and bottom right
        box_w = abs(self.bboxes[:, 0] - self.bboxes[:, 2])
        self.bboxes[:, 0] -= box_w
        self.bboxes[:, 2] += box_w

        # reverse the data in the second axis of the image data
        self.img = (self.img.numpy()[:, ::-1, :] * 255).astype(np.uint8)

        # return the reversed image and bounding boxes
        return self.img, self.bboxes.numpy()


def create_flipped_dataset(dataset, labels_ordered, torch_files_folder, resize, transform):

    flipped_input_file = torch_files_folder / 'flipped_input.pt'
    flipped_target_file = torch_files_folder / 'flipped_target.pt'

    # if no flipped target file already created
    if not os.path.isfile(flipped_target_file):

        flipped_inputs_list = []
        flipped_boxes_list = []

        # loop through dataset
        for i in range(len(dataset)):

            cur_obs = dataset[i]

            # flip the image
            flip_obs = HorizontalFlip(cur_obs[0], cur_obs[1])
            flipped_input, flipped_boxes = flip_obs()

            # append to list
            flipped_inputs_list.append(flipped_input)
            flipped_boxes_list.append(flipped_boxes)

        flipped_target_list = []
        cur_target_dict = {}

        # loop through flipped boxes
        for i in range(len(flipped_boxes_list)):

            # add labels and boxes to dictionary
            cur_target_dict['labels'] = labels_ordered[i]
            cur_target_dict['boxes'] = flipped_boxes_list[i]

            # append dictionary to list and reset dictionary
            flipped_target_list.append(cur_target_dict)
            cur_target_dict = {}

        # save flipped inputs and targets
        torch.save(flipped_inputs_list, flipped_input_file)
        torch.save(flipped_target_list, flipped_target_file)

    # if already flipped target file, load it and the flipped input file
    else:
        flipped_inputs_list = torch.load(flipped_input_file)

    # create flipped dataset object
    flipped_dataset = ObjectDetectionDataSet(flipped_inputs_list, False, flipped_target_file, 'neither', resize,
                                             transform=transform)

    return flipped_dataset

