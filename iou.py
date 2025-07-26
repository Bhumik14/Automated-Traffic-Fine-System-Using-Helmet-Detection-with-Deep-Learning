def iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute intersection area
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    interArea = inter_width * inter_height

    # Compute areas of the boxes
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))

    # Compute IoU with epsilon to avoid divide-by-zero
    union = boxAArea + boxBArea - interArea
    iou = interArea / (union + 1e-6) if union > 0 else 0.0

    return iou
