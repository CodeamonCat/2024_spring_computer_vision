import numpy as np
import pandas as pd
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def read_settings(setting_path: str) -> tuple[list, int, float]:
    with open(setting_path, "r", encoding="UTF-8") as f:
        setting = [line.rstrip('\n').split(',') for line in f.readlines()]
    f.close()
    return setting[1:6], int(setting[6][1]), float(setting[6][3])


def main():
    parser = argparse.ArgumentParser(
        description='main function of joint bilateral filter')
    parser.add_argument('--image_path',
                        default='./testdata/2.png',
                        help='path to input image')
    parser.add_argument('--setting_path',
                        default='./testdata/2_setting.txt',
                        help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ### TODO ###
    RGB_params, sigma_s, sigma_r = read_settings(args.setting_path)
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    img_BF = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    img_JBF = JBF.joint_bilateral_filter(img_rgb, img_gray)

    cost = dict()
    cost['BGR2GRAY'] = np.sum(
        np.abs(img_BF.astype('int32') - img_JBF.astype('int32')))
    # save image
    cv2.imwrite(
        f'./result/{args.image_path[-5:-4]}_image_cv2_COLOR_BGR2GRAY.png',
        img_gray)
    cv2.imwrite(f'./result/{args.image_path[-5:-4]}_BF.png', img_BF)
    cv2.imwrite(f'./result/{args.image_path[-5:-4]}_JBF.png', img_JBF)

    # 6 gray-scale images
    for R, G, B in RGB_params:
        # rgb to gray
        img_param_gray = img_rgb[:, :, 0] * float(
            R) + img_rgb[:, :, 1] * float(G) + img_rgb[:, :, 2] * float(B)
        img_param_JBF = JBF.joint_bilateral_filter(img_rgb, img_param_gray)
        cost[f'RGB_{R}_{G}_{B}'] = np.sum(
            np.abs(img_BF.astype('int32') - img_param_JBF.astype('int32')))
        img_param_JBF = cv2.cvtColor(img_param_JBF, cv2.COLOR_BGR2RGB)
        cv2.imwrite(
            f'./result/{args.image_path[-5:-4]}_RGB_{R}_{G}_{B}_gray.png',
            img_param_gray)
        cv2.imwrite(
            f'./result/{args.image_path[-5:-4]}_RGB_{R}_{G}_{B}_JBF.png',
            img_param_JBF)
    pd.Series(cost, name='Cost(1.pmg)').to_excel(
        f'./result/{args.image_path[-5:-4]}_cost_table.xlsx')


if __name__ == '__main__':
    main()
