# -*- coding: utf-8 -*-
import os
import numpy as np
from spectral import envi
from skimage import filters, morphology
from skimage.measure import label, regionprops

def load_hsi(raw_path, hdr_path):
    try:
        return envi.open(hdr_path, raw_path).load()
    except Exception:
        return None

def extract_foreground_mask(np_img):
    img_copy = np.copy(np_img)
    thr = np.percentile(img_copy, 99)
    img_copy[img_copy > thr] = thr
    
    band_100 = img_copy[:, :, 100]
    band_100[band_100 == 0] = np.nan
    band_ratio = np.nan_to_num(band_100, nan=0.0)
    
    try:
        thresh = filters.threshold_otsu(band_ratio)
        mask = morphology.binary_dilation(
            morphology.binary_erosion(band_ratio > thresh, morphology.square(3)), 
            morphology.square(3))
        return mask
    except Exception:
        return np.ones(np_img.shape[:2], dtype=bool)

def main():
    in_dir, out_dir = './', 'NPY_Mushroom_Dataset'
    os.makedirs(out_dir, exist_ok=True)

    for day in [1, 3, 5, 7, 9]:
        for y in range(1, 5):
            img_name = f"{day}-{y}-1"
            raw_path, hdr_path = os.path.join(in_dir, f"{img_name}_RT.raw"), os.path.join(in_dir, f"{img_name}_RT.hdr")
            if not os.path.exists(raw_path): continue

            np_img = load_hsi(raw_path, hdr_path)
            if np_img is None: continue

            props = regionprops(label(extract_foreground_mask(np_img), connectivity=2))
            valid_count = 0
            
            for prop in props:
                if not (1000 < prop.area < 50000): continue
                cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
                rs, re, cs, ce = cy - 75, cy + 75, cx - 75, cx + 75
                
                if rs < 0: re += -rs; rs = 0
                if re > np_img.shape[0]: rs -= (re - np_img.shape[0]); re = np_img.shape[0]
                if cs < 0: ce += -cs; cs = 0
                if ce > np_img.shape[1]: cs -= (ce - np_img.shape[1]); ce = np_img.shape[1]

                crop_img = np_img[rs:re, cs:ce, 14:240]
                if crop_img.shape == (150, 150, 226):
                    valid_count += 1
                    np.save(os.path.join(out_dir, f"Day_{day}_Group_{y}_Mushroom_{valid_count}.npy"), crop_img)

if __name__ == "__main__":
    main()