import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision import transforms


# create custom dataset object
class ObjectDetectionDataSet(Dataset):

    # initialize variables
    def __init__(self, inputs, inputs_path, targets, convert_to_format, resize, transform=None):
        self.inputs = inputs
        self.inputs_path = inputs_path
        self.targets = targets
        self.transform = transform
        self.convert_to_format = convert_to_format
        self.resize = resize

    # return length of item
    def __len__(self):
        return len(self.inputs)

    # get data in format to be sent to model
    def __getitem__(self, index):

        # get current input
        input_id = self.inputs[index]

        # get image data
        if self.inputs_path:
            x = self.read_images(input_id)
        else:
            x = np.array(Image.fromarray(input_id))

        # get and load current target
        target_id = self.targets
        y = torch.load(target_id)
        y = y[index]

        # if 4 channels, remove alpha channel
        if x.shape[-1] == 4:
            from skimage.color import rgba2rgb
            x = rgba2rgb(x)

        # get torch tensor of ground truth bounding boxes
        try:
            boxes = torch.from_numpy(y['boxes']).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y['boxes']).to(torch.float32)

        # get torch tensor of ground truth labels
        try:
            labels = torch.from_numpy(y['labels']).to(torch.int64)
        except TypeError:
            labels = torch.tensor(y['labels']).to(torch.int64)

        # convert to appropriate format
        if self.convert_to_format == 'xyxy':
            boxes = box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')
        elif self.convert_to_format == 'xywh':
            boxes = box_convert(boxes, in_fmt='xyxy', out_fmt='xywh')

        # target dictionary
        target = {'boxes': boxes,
                  'labels': labels}

        # check if any transformations
        if self.transform is not None:

            # make sure to resize bounding boxes if resizing image

            # get current height and width, and new height and width of image
            height = x.shape[0]
            width = x.shape[1]
            height_new = self.resize[0]
            width_new = self.resize[1]

            # for each bounding box
            new_bbox_list = []
            for box in boxes:

                # get current x and y for both points (or current x, y, w, and h)
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()

                # calculate new x and y for both points (or new x, y, w, and h)
                x1_new = round(x1 / width * width_new)
                y1_new = round(y1 / height * height_new)
                x2_new = round(x2 / width * width_new)
                y2_new = round(y2 / height * height_new)

                # append list of new bounding box coordinates to list
                # no need to worry about xyxy or xywh because it is all proportional
                new_bbox_list.append([x1_new, y1_new, x2_new, y2_new])

            # update bounding boxes to reflect resizing
            target['boxes'] = torch.tensor(new_bbox_list)

            # make image transformations
            x = self.transform(x)

        return x, target

    # read images in as np array
    @staticmethod
    def read_images(input_img):

        img_data = np.array(Image.open(input_img).convert('RGB'))

        return img_data


# custom collate function for the dataloader
def park_collate(batch):

    return tuple(zip(*batch))
