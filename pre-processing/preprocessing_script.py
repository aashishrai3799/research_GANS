import sys, os, cv2
import numpy as np

# This script serves as a demo of our image preprocessing pipeline for the
# CASIA-WebFace training set, as well as the LFW, CelebFacesA and Helen
# test sets.

# Note that all models were trained on RGB images, whereas openCV loads images
# in BGR channel order. The array[:, :, ::-1] indexing operation reverses this.

#############################33333############################################

# Parse facial landmarks from the celeba meta file
# returns a numpy array of [x, y] coordinates
def parse_celeba_landmarks(im_fn):
    fn = "list_landmarks_celeba.txt"
    with open(fn, "r") as f:
        lines = f.read().split("\n")[2:-1]

    parsed_lines = []
    for line in lines:
        lline = line.split()
        fn = lline[0]
        coords = list(map(int, lline[1:]))
        coords = np.array(coords).reshape((5, 2))
        parsed_lines.append((fn, coords))

    actual_fn = im_fn.split("/")[-1]
    img_ind = int(actual_fn.split(".")[0])
    return parsed_lines[img_ind - 1][1]

# Parse facial landmarks from the .pts files included with the Helen dataset
# returns a numpy array of [x, y] coordinates
def parse_helen_landmarks(fn):
    s = open(fn, "r").read().split("\n")[3:-1]
    pts = []
    for line in s:
        fLine = list(map(float, line.split(" ")))
        pts.append(fLine)
    pts = np.array(pts)
    return pts

# Crop a face image based on landmark coordinates
# Method outline: first, we calculate the center of the face as the mean of
# all landmark coordinates. Then, we crop a square centered around the face
# center, with a side 2.4 times as long as the maximal distance between the
# center and any individual landmark
def landmark_crop(im, pts):
    im = np.pad(im, [[4000, 4000], [4000, 4000], [0, 0]], mode="constant")
    pts += 4000
    centre = pts.mean(0, keepdims=True)
    dists = np.sqrt(np.square(pts - centre).sum(1))
    d = dists.max() * 1.2
    left   = int(round(centre[0, 0] - d))
    right  = int(round(centre[0, 0] + d))
    top    = int(round(centre[0, 1] - d))
    bottom = int(round(centre[0, 1] + d))
    im = im[top : bottom, left : right, :]
    return im

# CASIA and LFW: no landmarks, the images are 250*250 px and we just take the
# 192*192 px central crop
def casia_crop(fn):
    return cv2.imread(fn)[:,:, ::-1]

def lfw_crop(fn):
    return casia_crop(fn)

def celeba_crop(im_fn):
    landmarks = parse_celeba_landmarks(im_fn)
    im = cv2.imread(im_fn)[:, :, ::-1]

    return landmark_crop(im, landmarks)

def helen_crop(im_fn):
    pts_fn = im_fn.split(".")[0] + ".pts"
    landmarks = parse_helen_landmarks(pts_fn)
    im = cv2.imread(im_fn)[:, :, ::-1]
    return landmark_crop(im, landmarks)


# resampling function to derive the high-res and low-res reference images
# of 192px and 24px from face crops of arbitrary resoulution
def resample(im, dsize=192):
    ratio = im.shape[0] / dsize
    s = 0.25 * ratio
    filtered = cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=s, sigmaY=s)
    inter = cv2.INTER_CUBIC
    lr = cv2.resize(filtered, dsize=(dsize, dsize), interpolation=inter)
    return lr

if __name__ == "__main__":
    c = 0
    path = '/media/dl/DL/aashish/upscaling/images/'

    clas = os.listdir(str(path))
    clas = ['hr']
    for cl in clas:
        files = os.listdir(path + cl)
        for k in files:
            c+=1
            if c==5000:
                break
            img_fn = str(path + cl + '/' + k)
            img = lfw_crop(img_fn)
            hr = resample(img, dsize=192)
            lr = resample(hr, dsize=24)

            #cv2.imwrite("celeba_hr.png", hr[:, :, ::-1])
            cv2.imwrite("/media/dl/DL/aashish/upscaling/preprocess/lr/lfw_" + str(c) + '.jpg', lr[:, :, ::-1])
