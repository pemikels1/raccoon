import torch
from torchvision.ops import nms
from metrics import get_centroid, euclid_distance, get_closest_boxes, get_iou


# training function
def train(model, epoch, optim, dataloader, device):

    # set model to train mode
    model.train()

    # loop through batch
    for batch_idx, (imgs, targets) in enumerate(dataloader):

        # convert imgs and targets to proper format for model and put on device
        imgs = list(imgs)
        imgs = [img.to(device) for img in imgs]
        targets = list(targets)
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

        # zero out the optimizer's gradient
        optim.zero_grad()

        # pass data through model and calculate the loss
        loss_dict = model(imgs, targets)
        cur_loss = sum(loss for loss in loss_dict.values())

        # back prop
        cur_loss.backward()

        # take optimizer step
        optim.step()

        print('Train Epoch: {} [{} / {} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(imgs), len(dataloader.dataset),
            100. * batch_idx / len(dataloader), cur_loss.item()))


# evaluate the model performance on a dataset. Currently done sequentially, plan to implement batch in future
def test(model, epoch, dataset, device, which_set, score_threshold=0.5, nms_thresh=0.1):
    # set model to evaluation mode
    model.eval()

    # ignore the gradient
    with torch.no_grad():

        # initialize list of calculated IOUs
        dataset_iou = []

        # for (imgs, targets) in range(len(dataset)):
        for idx, (imgs, targets) in enumerate(dataset):

            # put images on device and run through model
            imgs = imgs.to(device)
            output = model([imgs])

            # perform non-max suppression to get final predictions
            final_box_indices = nms(output[0]['boxes'], output[0]['scores'], nms_thresh)

            # get the ground truth boxes and their respective centroids
            gt_boxes = targets['boxes'].cpu().numpy()
            gt_centroids = [get_centroid(box) for box in gt_boxes]

            gt_boxes_used = []
            # for each prediction
            for box_idx in final_box_indices:

                # filter by prediction score threshold
                if output[0]['scores'][box_idx] > score_threshold:

                    # get the bounding box coordinates and centroid of the prediction
                    cur_coords = output[0]['boxes'][box_idx]
                    cur_centroid = get_centroid(cur_coords)

                    # choose the closest ground truth box for the prediction and calculate IOU between the two
                    closest_box = get_closest_boxes(gt_centroids, cur_centroid)

                    # don't want to use ground truth twice in IOU evaluation
                    if closest_box not in gt_boxes_used:

                        # keep track of ground truths already used in IOU evaluation
                        gt_boxes_used.append(closest_box)
                        iou = get_iou(cur_coords, gt_boxes[closest_box])
                        dataset_iou.append(iou)

    # average the IOUs for the entire dataset
    new_dataset_iou = [iou.item() for iou in dataset_iou]
    avg_iou = sum(new_dataset_iou) / (len(new_dataset_iou))

    print('\nEvaluating Epoch {} on {} set: Average IOU: {:.1f}%)\n'.format(
        epoch, which_set, avg_iou * 100))

    return avg_iou, new_dataset_iou

