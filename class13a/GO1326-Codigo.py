# GO1326-Codigo
iou_thresholds = [0.5, 0.55, 0.6, ..., 0.95]  # 10 valores

mAPs = []
for iou_thresh in iou_thresholds:
    mAP = calculate_mAP(predictions, groundtruth, iou_thresh)
    mAPs.append(mAP)

mAP_50_95 = np.mean(mAPs)
