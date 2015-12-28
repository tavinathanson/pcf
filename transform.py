from os import path
import pandas as pd
from skimage.data import imread
from skimage.io import imshow
import numpy as np

training = pd.read_csv("training.csv", 
                       header=None,
                       names=['name', 'fga'],
                       dtype={'name': object, 'fga': float})

def transform_pcf(patch_size=512, limit=None, partial_save=False):
    training = pd.read_csv("training.csv", 
                           header=None,
                           names=['name', 'fga'],
                           dtype={'name': object, 'fga': float})
    if limit:
        training = training.head(limit)
    from skimage.io import imsave
    all_patches = []
    output = []
    for i, row in training.iterrows():
        row_patches = []
        row_output = []
        name = row['name']
        fga = row['fga']
        print("Transforming image %s" % name)
        import sys
        sys.stdout.flush()
        for kind in ["DX", "TS"]:
            img = imread("images/%s/%s-%s.png" % (kind, name, kind))
            img_patches = transform_img(img, name, patch_size=patch_size)
            for img_patch in img_patches:
                for rot_i in range(4):
                    all_patches.append(np.rot90(img_patch, k=rot_i))
                    row_patches.append(np.rot90(img_patch, k=rot_i))
                    output.append(fga)
                    row_output.append(fga)
        X_row = np.asarray(row_patches)
        y_row = np.asarray(row_output)
        if partial_save:
            with open("X_file_%s" % name, 'w') as f:
                np.save(f, X_row)
            with open("y_file_%s" % name, 'w') as f:
                np.save(f, y_row)
    X = np.asarray(all_patches)
    y = np.asarray(output)
    with open("X_file", 'w') as f:
        np.save(f, X)
    with open("y_file", 'w') as f:
        np.save(f, y)
    return X, y

def transform_img(img, name, patch_size):
    from skimage.transform import rotate
    from skimage.exposure import is_low_contrast
    rows, cols, _ = img.shape
    max_dim = max(rows, cols)
    # If more rows than cols, vertical
    vertical = True if rows == max_dim else False
    if rows < patch_size or cols < patch_size:
        print("Image too small: %s-%s.png" % (name))
        return []
    # Make all images vertical for now (more rows than cols)
    if not vertical:
        img = img.transpose((1, 0, 2))
        rows, cols, _ = img.shape
    row_blocks = rows / patch_size
    col_blocks = cols / patch_size
    patches = []
    for i in range(row_blocks):
        for j in range(col_blocks):
            patch = img[patch_size * i:patch_size * (i + 1),
                        patch_size * j:patch_size * (j + 1),
                        :]
            if not is_low_contrast(patch):
                patches.append(patch)
    return patches

transform_pcf()
