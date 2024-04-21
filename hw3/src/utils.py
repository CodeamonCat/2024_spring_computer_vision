import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2 * N, 9))
    for i in range(N):
        A[i * 2, :] = [
            u[i][0], u[i][1], 1, 0, 0, 0, -u[i][0] * v[i][0],
            -u[i][1] * v[i][0], -v[i][0]
        ]
        A[i * 2 + 1, :] = [
            0, 0, 0, u[i][0], u[i][1], 1, -u[i][0] * v[i][1],
            -u[i][1] * v[i][1], -v[i][1]
        ]

    # TODO: 2.solve H with A
    U, S, Vh = np.linalg.svd(A)
    h = Vh.T[:, -1]
    H = h.reshape((3, 3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    U_x, U_y = np.meshgrid(range(xmin, xmax), range(ymin, ymax))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    U = np.concatenate(
        ([U_x.reshape(-1)], [U_y.reshape(-1)
                             ], [np.ones((xmax - xmin) * (ymax - ymin))]),
        axis=0)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H_inv, U)
        V = np.divide(V, V[2])
        V_x = V[0].reshape(ymax - ymin, xmax - xmin)
        V_y = V[1].reshape(ymax - ymin, xmax - xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask_x = (0 <= V_x) & (V_x < w_src - 1)
        mask_y = (0 <= V_y) & (V_y < h_src - 1)
        mask = mask_x & mask_y

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        masked_V_x = V_x[mask]
        masked_V_y = V_y[mask]

        # bilinear interpolation
        masked_V_x_int = masked_V_x.astype(int)
        masked_V_y_int = masked_V_y.astype(int)
        d_V_x = (masked_V_x - masked_V_x_int).reshape((-1, 1))
        d_V_y = (masked_V_y - masked_V_y_int).reshape((-1, 1))
        target = np.zeros(src.shape)
        target[masked_V_y_int, masked_V_x_int] += (1 - d_V_y) * (
            1 - d_V_x) * src[masked_V_y_int, masked_V_x_int]
        target[masked_V_y_int,
               masked_V_x_int] += d_V_y * (1 - d_V_x) * src[masked_V_y_int + 1,
                                                            masked_V_x_int]
        target[masked_V_y_int,
               masked_V_x_int] += (1 - d_V_y) * d_V_x * src[masked_V_y_int,
                                                            masked_V_x_int + 1]
        target[masked_V_y_int,
               masked_V_x_int] += d_V_y * d_V_x * src[masked_V_y_int + 1,
                                                      masked_V_x_int + 1]

        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax, xmin:xmax][mask] = target[masked_V_y_int,
                                                 masked_V_x_int]
        # pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        V = np.dot(H, U)
        V = np.divide(V, V[2]).astype(int)
        V_x = V[0].reshape(ymax - ymin, xmax - xmin)
        V_y = V[1].reshape(ymax - ymin, xmax - xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask_x = (0 <= V_x) & (V_x < w_dst)
        mask_y = (0 <= V_y) & (V_y < h_dst)
        mask = mask_x & mask_y

        # TODO: 5.filter the valid coordinates using previous obtained mask
        masked_V_x = V_x[mask]
        masked_V_y = V_y[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[masked_V_y, masked_V_x, :] = src[mask]
        # pass

    return dst
