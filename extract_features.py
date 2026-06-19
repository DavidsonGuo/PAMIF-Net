# -*- coding: utf-8 -*-
"""
Multi-dimensional Feature Extraction Pipeline for Hyperspectral Imaging
Extracts PCA, CNN, MGAF, Patch, Color, and Texture features.
"""

import os, warnings
import numpy as np, pandas as pd, h5py
from spectral import envi
from skimage import filters, morphology
from skimage.measure import label, regionprops
from skimage.transform import resize
from skimage.color import rgb2gray, rgb2lab
from skimage.exposure import rescale_intensity
from skimage.feature import local_binary_pattern, gabor
from skimage.measure import shannon_entropy
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
from keras import layers, models

try:
    from skimage.feature import greycomatrix, greycoprops
    GLCM_AVAILABLE = True
except ImportError:
    import mahotas
    GLCM_AVAILABLE = False
    warnings.warn("skimage.greycomatrix not available, using mahotas fallback.")

def load_hsi(raw_path, hdr_path):
    return envi.open(hdr_path, raw_path).load()

def remove_high_reflectivity_noise(np_img):
    threshold = np.percentile(np_img, 99)
    np_img[np_img > threshold] = threshold
    return np_img

def segment_objects(img):
    np_img = remove_high_reflectivity_noise(np.array(img))
    band = np_img[:, :, 100].copy()
    band[band == 0] = np.nan
    band = np.nan_to_num(band, nan=0.0)
    try:
        return band > filters.threshold_otsu(band)
    except ValueError:
        return None

def clean_noise(binary_image, kernel_size=1):
    med = filters.median(binary_image, morphology.disk(kernel_size))
    sq1 = morphology.square(1)
    # Replicates original logic exactly but simplified
    return morphology.binary_dilation(morphology.binary_erosion(med, sq1), sq1)

def apply_pca(np_img, n_components):
    pca = PCA(n_components=n_components)
    pca_img = pca.fit_transform(np_img.reshape(-1, np_img.shape[2]))
    return pca_img.reshape(np_img.shape[0], np_img.shape[1], -1), pca

def label_samples(cleaned_img):
    labeled, _ = label(cleaned_img, return_num=True, background=0)
    props = [p for p in regionprops(labeled) if p.eccentricity < 0.99 and 300 < p.area < 550000]
    if not props: return labeled, 0, []

    centroids = np.array([p.centroid for p in props])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(centroids[:, 1].reshape(-1, 1))
    
    col_centers = [np.mean(centroids[kmeans.labels_ == i, 1]) for i in range(2)]
    col_order = np.argsort(col_centers)
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(col_order)}
    
    for i, p in enumerate(props): p.col_idx = mapping[kmeans.labels_[i]]
    
    sorted_labels = np.zeros_like(labeled, dtype=int)
    info, counter = [], 1
    for col in range(2):
        col_props = sorted([p for p in props if p.col_idx == col], key=lambda x: -x.centroid[0])
        for p in col_props:
            sorted_labels[labeled == p.label] = counter
            info.append({"ID": counter, "Pixel Size": p.area, "Width": p.bbox[3]-p.bbox[1], "Height": p.bbox[2]-p.bbox[0]})
            counter += 1
            
    return sorted_labels, len(props), info

def calc_avg_spectrum(np_img, labels, num_samples):
    spectra = {}
    for sid in range(1, num_samples + 1):
        mask = (labels == sid)
        spectra[sid] = [np_img[:, :, b][mask].mean() if np.any(mask) else np.nan for b in range(np_img.shape[2])]
    return spectra

def extract_color_texture(np_img, labels, prefix):
    R, G, B = np_img[:,:,171], np_img[:,:,114], np_img[:,:,57]
    records = []
    
    for sid in [i for i in np.unique(labels) if i != 0]:
        mask = labels == sid
        feat = {"ID": sid}
        
        rgb_crop = np.stack([R, G, B], axis=2)
        for band, lbl in zip([R, G, B], ['R', 'G', 'B']):
            md = band[mask]
            feat.update({f"Mean_{lbl}": np.mean(md), f"Std_{lbl}": np.std(md),
                         f"Skew_{lbl}": skew(md.flatten()), f"Kurt_{lbl}": kurtosis(md.flatten()),
                         f"Entropy_{lbl}": shannon_entropy(md)})
        
        rgb_norm = (rgb_crop - np.min(rgb_crop)) / (np.max(rgb_crop) - np.min(rgb_crop) + 1e-6)
        lab = rgb2lab(rgb_norm)
        for ch, lbl in zip([lab[:,:,0], lab[:,:,1], lab[:,:,2]], ['L', 'a', 'b']):
            feat.update({f"Mean_{lbl}": np.mean(ch[mask]), f"Std_{lbl}": np.std(ch[mask])})
        
        gray = rescale_intensity(rgb2gray(rgb_norm) * mask, in_range='image', out_range=(0, 255)).astype(np.uint8)
        if GLCM_AVAILABLE:
            glcm = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
            feat.update({f"Texture_{k}": greycoprops(glcm, k.lower())[0,0] for k in ['Contrast', 'Correlation', 'Energy', 'Homogeneity']})
        else:
            try:
                hf = mahotas.features.haralick(gray).mean(axis=0)
                feat.update({"Texture_Contrast": hf[1], "Texture_Correlation": hf[2], "Texture_Energy": hf[8], "Texture_Homogeneity": hf[4]})
            except:
                feat.update({f"Texture_{k}": np.nan for k in ['Contrast', 'Correlation', 'Energy', 'Homogeneity']})
                
        for theta, tlbl in zip([0, np.pi/4, np.pi/2, 3*np.pi/4], ['0', '45', '90', '135']):
            filt_real, _ = gabor(gray, frequency=0.6, theta=theta)
            feat.update({f"Gabor{tlbl}_mean": np.mean(filt_real[mask]), f"Gabor{tlbl}_std": np.std(filt_real[mask])})
            
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        feat.update({"LBP_mean": np.mean(lbp[mask]), "LBP_std": np.std(lbp[mask])})
        records.append(feat)
        
    pd.DataFrame(records).to_csv(f"{prefix}_color_texture.csv", index=False)

class FeatureExtractor(models.Model):
    def __init__(self):
        super().__init__()
        self.seq = models.Sequential([
            layers.Conv3D(32, (9,9,3), padding='same'),
            layers.MaxPooling3D((3,3,5)),
            layers.Conv3D(64, (5,5,1), padding='same'),
            layers.MaxPooling3D((5,5,2)),
            layers.Reshape((10, 10, 64)),
            layers.Conv2D(64, (5,5), padding='same'),
            layers.Conv2D(32, (3,3), padding='same'),
            layers.Reshape((100, 32))
        ])
    def call(self, x): return self.seq(x)

def extract_save_cnn(pca_img, labels, fname):
    model = FeatureExtractor()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    with h5py.File(fname, 'w') as h5f:
        for sid in [i for i in np.unique(labels) if i != 0]:
            pixels = pca_img[labels == sid]
            curr_len, target_len = len(pixels) * 10, 150 * 150 * 10
            
            if curr_len < target_len:
                pad = np.full(((target_len - curr_len) // 10, 10), np.mean(pixels[pixels > 0]))
                pixels = np.vstack((pixels, pad))
            elif curr_len > target_len:
                pixels = pixels[:target_len // 10, :]
                
            feat = np.squeeze(model.predict(pixels.reshape(1, 150, 150, 10, 1), verbose=0), axis=0)
            h5f.create_dataset(f'features_{sid}', data=feat)

def compute_mgaf(curve, sigma=2.0):
    b = len(curve)
    if np.any(np.isnan(curve)) or np.all(curve == curve[0]): return np.zeros((b, b))
    x_hat = np.nan_to_num((curve - np.nanmean(curve)) / (np.nanstd(curve) + 1e-6)).reshape(-1, 1)
    W = np.exp(-((np.arange(b).reshape(-1,1) - np.arange(b).reshape(1,-1))**2) / (2 * sigma**2))
    return np.nan_to_num(W * (x_hat @ x_hat.T - (1 - x_hat**2) @ (1 - x_hat.T**2)) ** 0.5)

def extract_save_patch(labels, fname, np_img, pca_model):
    all_p, all_m = [], []
    for sid in [i for i in np.unique(labels) if i != 0]:
        coords = np.argwhere(labels == sid)
        if coords.shape[1] != 2: continue
        
        rmin, cmin = coords.min(axis=0)
        rmax, cmax = coords.max(axis=0)
        crop = resize(np_img[rmin:rmax+1, cmin:cmax+1, :], (150, 150, np_img.shape[2]), preserve_range=True, anti_aliasing=True)
        
        p_list, m_list = [], []
        for r in range(0, 150, 15):
            for c in range(0, 150, 15):
                patch = crop[r:r+15, c:c+15, :]
                if patch.shape[:2] != (15, 15): continue
                spec = pca_model.transform(np.mean(patch.reshape(-1, patch.shape[2]), axis=0).reshape(1, -1)).flatten()
                p_list.append(spec)
                m_list.append(compute_mgaf(spec))
                
        all_p.append(np.stack(p_list))
        all_m.append(np.stack(m_list))
        
    with h5py.File(fname.replace('.h5', '_patch.h5'), 'w') as f: f.create_dataset('patch_features', data=np.stack(all_p))
    with h5py.File(fname.replace('.h5', '_mgaf.h5'), 'w') as f: f.create_dataset('mgaf_features', data=np.stack(all_m))

def main():
    directory = './'
    for i in range(1, 5):
        for j in range(1, 5):
            base = f'5-{i}-{j}'
            raw_p, hdr_p = os.path.join(directory, f'{base}_RT.raw'), os.path.join(directory, f'{base}_RT.hdr')
            if not os.path.exists(raw_p): continue
            
            img = load_hsi(raw_p, hdr_p)[50:-250, 250:1000, 14:, ::-1, :]
            mask = segment_objects(img)
            if mask is None: continue
            
            pca_img, pca_model = apply_pca(img, 10)
            labels, num, info = label_samples(clean_noise(mask))
            
            pd.DataFrame.from_dict(calc_avg_spectrum(img, labels, num), orient='index').to_csv(f'{base}_avg_spectra.csv', header=False)
            pd.DataFrame(info).to_csv(f'{base}_dimensions.csv', index=False)
            extract_color_texture(img, labels, base)
            extract_save_cnn(pca_img, labels, f'{base}_CNN.h5')
            extract_save_patch(labels, f'{base}.h5', img, pca_model)

if __name__ == "__main__":
    main()