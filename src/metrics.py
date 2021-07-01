# get centroid of bounding box
def get_centroid(box):

    centroid_x = (box[0] + box[2]) / 2
    centroid_y = (box[1] + box[3]) / 2

    return [centroid_x, centroid_y]


# Euclidean distance between two points
def euclid_distance(v1, v2):

    return sum((p-q) ** 2 for p, q in zip(v1, v2)) ** 0.5


# find closest ground truth bounding box for each prediction bounding box
def get_closest_boxes(gt_boxes, pred_box):

    min_distance = float('inf')
    min_idx = -1

    # loop through ground truths
    for j in range(len(gt_boxes)):

        # calculate distance between current ground truth bounding box and current prediction bounding box
        cur_distance = euclid_distance(gt_boxes[j], pred_box).item()

        # if current distance less than min distance, replace it and keep track of position
        if cur_distance < min_distance:
            min_distance = cur_distance
            min_idx = j

    return min_idx


def get_iou(box_1, box_2):

    # get left-most x value of right box
    box_1_x_1 = box_1[0]
    box_2_x_1 = box_2[0]
    x_a = max(box_1_x_1, box_2_x_1)

    # get highest y value of bottom box
    box_1_y_1 = box_1[1]
    box_2_y_1 = box_2[1]
    y_a = max(box_1_y_1, box_2_y_1)

    # get right-most x value of left box
    box_1_x_2 = box_1[2]
    box_2_x_2 = box_2[2]
    x_b = min(box_1_x_2, box_2_x_2)

    # get lowest y value of top box
    box_1_y_2 = box_1[3]
    box_2_y_2 = box_2[3]
    y_b = min(box_1_y_2, box_2_y_2)

    # calculate the intersection area
    # max 0 is in case the boxes do not overlap
    intersection_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    # calculate each box's area
    box_a_area = (box_1_x_2 - box_1_x_1) * (box_1_y_2 - box_1_y_1)
    box_b_area = (box_2_x_2 - box_2_x_1) * (box_2_y_2 - box_2_y_1)

    # calculate and return the intersection over union
    iou = intersection_area / (box_a_area + box_b_area - intersection_area)

    return iou
