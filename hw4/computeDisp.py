import numpy as np
import cv2
import cv2.ximgproc as xip


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    cost_left = np.zeros((8, h, w, ch), dtype=int)
    cost_right = np.zeros((8, h, w, ch), dtype=int)
    padded_Il = np.pad(Il, pad_width=1, mode='constant',
                       constant_values=0)[:, :, 1:-1]
    padded_Ir = np.pad(Ir, pad_width=1, mode='constant',
                       constant_values=0)[:, :, 1:-1]

    kernel = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
              (1, 1)]

    for idx, (dr, dc) in enumerate(kernel):
        shifted_Il = np.roll(padded_Il, shift=(dr, dc), axis=(0, 1))
        cost_left[idx] = (shifted_Il[1:-1, 1:-1] < Il)
        shifted_Ir = np.roll(padded_Ir, shift=(dr, dc), axis=(0, 1))
        cost_right[idx] = (shifted_Ir[1:-1, 1:-1] < Ir)

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)

    cost_volume_left = np.zeros((max_disp + 1, h, w))
    cost_volume_right = np.zeros((max_disp + 1, h, w))

    for disp in range(max_disp + 1):
        shift_left = cost_left[:, :, disp:].astype(np.uint32)
        shift_right = cost_right[:, :, :w - disp].astype(np.uint32)
        # sum and calcualte Hamming distance with XOR on all channels
        Hd_cost = np.sum(shift_left ^ shift_right, axis=0)
        raw_cost = np.sum(Hd_cost, axis=2).astype(np.float32)

        smoothed_raw_left = cv2.copyMakeBorder(raw_cost, 0, 0, disp, 0,
                                               cv2.BORDER_REPLICATE)
        smoothed_raw_right = cv2.copyMakeBorder(raw_cost, 0, 0, 0, disp,
                                                cv2.BORDER_REPLICATE)

        cost_volume_left[disp] = xip.jointBilateralFilter(Il,
                                                          smoothed_raw_left,
                                                          d=-1,
                                                          sigmaColor=4,
                                                          sigmaSpace=9)
        cost_volume_right[disp] = xip.jointBilateralFilter(Ir,
                                                           smoothed_raw_right,
                                                           d=-1,
                                                           sigmaColor=4,
                                                           sigmaSpace=9)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all

    disp_map_left = np.argmin(cost_volume_left, axis=0)
    disp_map_right = np.argmin(cost_volume_right, axis=0)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering

    # Left-right consistency check
    consistency = np.zeros((h, w), dtype=np.float32)
    x, y = np.meshgrid(range(w), range(h))

    temp_x = x - disp_map_left
    valid_coords = (temp_x >= 0)
    disp_left_valid = disp_map_left[valid_coords]
    disp_right_valid = disp_map_right[y[valid_coords], temp_x[valid_coords]]
    consistency_matches = (disp_left_valid == disp_right_valid)

    consistency[y[valid_coords][consistency_matches],
                x[valid_coords][consistency_matches]] = disp_left_valid[
                    consistency_matches]

    # Hole filling
    padded_consistency = cv2.copyMakeBorder(consistency,
                                            0,
                                            0,
                                            1,
                                            1,
                                            cv2.BORDER_CONSTANT,
                                            value=max_disp)
    labels_left = np.zeros((h, w), dtype=np.float32)
    labels_right = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            index_left, index_right = 0, 0
            while padded_consistency[y, x + 1 - index_left] == 0:
                index_left += 1
            labels_left[y, x] = padded_consistency[y, x + 1 - index_left]
            while padded_consistency[y, x + 1 + index_right] == 0:
                index_right += 1
            labels_right[y, x] = padded_consistency[y, x + 1 + index_right]

    labels = np.min((labels_left, labels_right), axis=0)
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, 12)

    return labels.astype(np.uint8)
