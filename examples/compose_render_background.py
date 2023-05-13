import cv2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--back', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    back = cv2.imread(args.back)
    img = cv2.imread(args.path, -1)
    assert back.shape[:2] == img.shape[:2], (img.shape, back.shape)

    alpha = img[:, :, 3:] / 255.
    compose = img[..., :3] * (alpha) + back * (1 - alpha)
    cv2.imwrite('debug.jpg', compose)