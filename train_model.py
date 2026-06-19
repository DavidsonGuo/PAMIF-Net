# -*- coding: utf-8 -*-
"""
PAMIF-Net Training and Ablation Framework
- Five-class prediction: {1,3,5,7,9} mapped to {0,1,2,3,4}
- 10 independent stratified trials (Train/Val/Test)
- Gated multimodal fusion with CNN, PATCH, MGAF, SPEC, and COLOR branches.
- FiLM modulation conditioned on tabular features.
"""

import os, time, json, h5py, warnings
from typing import Dict, List
import numpy as np, pandas as pd
import tensorflow as tf
from keras import layers, callbacks, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# ==========================================
# 1. Utilities and Metrics
# ==========================================
def load_h5_any(path: str) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        ks = list(f.keys())
        assert ks, f"No datasets in {path}"
        if len(ks) == 1: return f[ks[0]][:]
        ks = sorted(ks, key=lambda k: int(''.join(filter(str.isdigit, k)) or 0))
        return np.stack([f[k][:] for k in ks], axis=0)

def std_channelwise(X: np.ndarray, ch_axis=-1):
    axes = tuple(i for i in range(X.ndim) if i != ch_axis)
    return (X - X.mean(axis=axes, keepdims=True)) / (X.std(axis=axes, keepdims=True) + 1e-6)

def map_labels_robust(y_raw: np.ndarray) -> np.ndarray:
    y = np.array(y_raw).reshape(-1).astype(int)
    s = set(np.unique(y).tolist())
    if s == {1, 3, 5, 7, 9}: mp = {1:0, 3:1, 5:2, 7:3, 9:4}
    else: mp = {v: i for i, v in enumerate(sorted(s))}
    return np.array([mp[v] for v in y], dtype=np.int64)

def days_mae(y_true, y_pred):
    to_days = lambda v: v * 2 + 1
    return float(np.mean(np.abs(to_days(y_true) - to_days(y_pred))))

def save_cm_tif(y_true, y_pred, path, names=('1','3','5','7','9'), dpi=300):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    fig, ax = plt.subplots(figsize=(5,4), dpi=dpi)
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(5), yticks=range(5), xticklabels=names, yticklabels=names, 
           xlabel='Predicted', ylabel='True', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(path, format='tif', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

class QWK_Cls(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name="qwk", **kw):
        super().__init__(name=name, **kw)
        self.K = num_classes
        self.cm = self.add_weight("cm", shape=(self.K, self.K), initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_hat = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true_cls = tf.argmax(y_true, axis=-1, output_type=tf.int32) if y_true.shape.rank > 1 else tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        self.cm.assign_add(tf.cast(tf.math.confusion_matrix(y_true_cls, y_hat, num_classes=self.K), tf.float32))

    def result(self):
        n = tf.reduce_sum(self.cm)
        def _compute():
            O = self.cm / (n + 1e-8)
            r, c = tf.reduce_sum(O, axis=1), tf.reduce_sum(O, axis=0)
            idx = tf.range(self.K, dtype=tf.float32)
            W = tf.square(idx[:, None] - idx[None, :]) / tf.square(tf.cast(self.K - 1, tf.float32))
            E = tf.tensordot(r, c, axes=0)
            return 1.0 - tf.reduce_sum(W * O) / (tf.reduce_sum(W * E) + 1e-8)
        return tf.cond(tf.equal(n, 0), lambda: tf.constant(0.0, tf.float32), _compute)

    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))

# ==========================================
# 2. Data Loading
# ==========================================
def load_dataset(pcnn='X_CNN.h5', ppatch='X_PATCH.h5', pmgaf='X_MGAF.h5', pspec='X_spectral.csv', pcolor='X_color.csv', py='y.h5'):
    X_cnn = load_h5_any(pcnn)
    cnn_mode = '2d' if X_cnn.ndim == 4 else '1d'
    X_cnn = std_channelwise(X_cnn)
    
    Xp = std_channelwise(np.transpose(load_h5_any(ppatch).astype(np.float32)[..., None], (0, 1, 3, 2, 4)))
    Xm = std_channelwise(np.transpose(load_h5_any(pmgaf).astype(np.float32), (0, 1, 3, 2, 4)))
    Xs = pd.read_csv(pspec, header=None).values.astype(np.float32)
    Xc = pd.read_csv(pcolor, header=None).values.astype(np.float32)
    y = map_labels_robust(load_h5_any(py).reshape(-1))

    print(f"Data loaded - CNN: {X_cnn.shape} ({cnn_mode}), Patch: {Xp.shape}, MGAF: {Xm.shape}, Spec: {Xs.shape}, Color: {Xc.shape}, Y: {y.shape}")
    return {'X_cnn': X_cnn, 'cnn_mode': cnn_mode, 'cnn_shape': X_cnn.shape[1:],
            'X_patch': Xp, 'patch_shape': Xp.shape[1:], 'X_mgaf': Xm, 'mgaf_shape': Xm.shape[1:],
            'X_spec': Xs, 'spec_len': Xs.shape[1], 'X_color': Xc, 'y': y}

# ==========================================
# 3. Model Architecture Components
# ==========================================
def SE_block(x, mode="2d", ratio=4):
    ch = x.shape[-1]
    if mode == "3d": gap = layers.GlobalAveragePooling3D()(x); rs = (1, 1, 1, ch)
    elif mode == "1d": gap = layers.GlobalAveragePooling1D()(x); rs = (1, ch)
    else: gap = layers.GlobalAveragePooling2D()(x); rs = (1, 1, ch)
    
    s = layers.Dense(max(ch // ratio, 8), activation='relu')(gap)
    s = layers.Dense(ch, activation='sigmoid')(s)
    return layers.Multiply()([x, layers.Reshape(rs)(s)])

def conv_block(x, f, k=3, mode="2d"):
    conv_layer = layers.Conv3D if mode == "3d" else (layers.Conv2D if mode == "2d" else layers.Conv1D)
    x = conv_layer(f, k, padding='same', use_bias=False)(x)
    return layers.Activation('relu')(layers.BatchNormalization()(x))

def gated_fuse(tokens: List[tf.Tensor], d_model=128, dr=0.2):
    if len(tokens) == 1: x = tokens[0]
    else:
        T = tf.stack(tokens, axis=1)
        w = tf.nn.softmax(layers.Dense(1)(T), axis=1)
        x = tf.reduce_sum(T * w, axis=1)
    return layers.Dropout(dr)(layers.Dense(d_model, activation='relu')(x))

def apply_film(feat_map, cond_vec, mode):
    ch = int(feat_map.shape[-1])
    rs = (1, ch) if mode == "1d" else ((1, 1, ch) if mode == "2d" else (1, 1, 1, ch))
    gamma = layers.Reshape(rs)(layers.Dense(ch, activation='tanh')(cond_vec))
    beta = layers.Reshape(rs)(layers.Dense(ch, activation='tanh')(cond_vec))
    return layers.Add()([layers.Multiply()([feat_map, gamma]), beta])

def build_pamif_net(cnn_shape, cnn_mode, patch_shape=None, mgaf_shape=None, spec_len=None, color_dim=None, 
                    flags=(1,1,1,1,1), d_model=128, dr=0.2, K=5):
    use_cnn, use_patch, use_mgaf, use_spec, use_color = flags
    inputs, tokens, cond_vec = [], [], None

    if use_cnn:
        inp_cnn = layers.Input(shape=cnn_shape, name='inp_cnn'); inputs.append(inp_cnn)
        cnn_fm = SE_block(conv_block(conv_block(inp_cnn, 64, mode=cnn_mode), 128, mode=cnn_mode), mode=cnn_mode)

    if use_spec:
        inp_spec = layers.Input(shape=(spec_len,), name='inp_spec'); inputs.append(inp_spec)
        spec_tok = layers.Dense(d_model, activation='relu')(layers.Dense(256, activation='relu')(layers.BatchNormalization()(inp_spec)))
        tokens.append(spec_tok); cond_vec = spec_tok if cond_vec is None else layers.Concatenate()([cond_vec, spec_tok])

    if use_color:
        inp_color = layers.Input(shape=(color_dim,), name='inp_color'); inputs.append(inp_color)
        color_tok = layers.Dense(d_model, activation='relu')(layers.Dense(128, activation='relu')(layers.BatchNormalization()(inp_color)))
        tokens.append(color_tok); cond_vec = color_tok if cond_vec is None else layers.Concatenate()([cond_vec, color_tok])

    if use_cnn:
        if cond_vec is not None: cnn_fm = apply_film(cnn_fm, cond_vec, mode=cnn_mode)
        gap = layers.GlobalAveragePooling2D()(cnn_fm) if cnn_mode == '2d' else layers.GlobalAveragePooling1D()(cnn_fm)
        tokens.append(layers.Dense(d_model, activation='relu')(gap))

    if use_patch:
        inp_patch = layers.Input(shape=patch_shape, name='inp_patch'); inputs.append(inp_patch)
        p_fm = SE_block(conv_block(conv_block(inp_patch, 32, mode='3d'), 64, mode='3d'), mode='3d')
        tokens.append(layers.Dense(d_model, activation='relu')(layers.GlobalAveragePooling3D()(p_fm)))

    if use_mgaf:
        inp_mgaf = layers.Input(shape=mgaf_shape, name='inp_mgaf'); inputs.append(inp_mgaf)
        m_fm = SE_block(conv_block(conv_block(inp_mgaf, 32, mode='3d'), 64, mode='3d'), mode='3d')
        tokens.append(layers.Dense(d_model, activation='relu')(layers.GlobalAveragePooling3D()(m_fm)))

    assert tokens, "At least one modality must be active."
    logits = layers.Dense(K, name='cls_logits')(gated_fuse(tokens, d_model, dr))
    return tf.keras.Model(inputs=inputs, outputs=logits, name="PAMIF_Net")

# ==========================================
# 4. Training and Evaluation Routine
# ==========================================
class EpochLogger(callbacks.Callback):
    def __init__(self, Xtr, Ytr, Xva, Yva, out_csv):
        super().__init__()
        self.Xtr, self.Ytr, self.Xva, self.Yva, self.out, self.rows = Xtr, Ytr, Xva, Yva, out_csv, []
        
    def on_epoch_begin(self, epoch, logs=None): self.t0 = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        row = {'epoch': epoch + 1, 'sec': float(time.time() - self.t0), 'loss': logs.get('loss'), 'val_loss': logs.get('val_loss')}
        def _eval(X, Y):
            pr = self.model.predict(X, verbose=0).argmax(axis=1)
            p, r, f1, _ = precision_recall_fscore_support(Y, pr, average='macro', zero_division=0)
            return accuracy_score(Y, pr), p, r, f1
        ta, tp, tr, tf1 = _eval(self.Xtr, self.Ytr)
        va, vp, vr, vf1 = _eval(self.Xva, self.Yva)
        row.update({'acc': ta, 'prec': tp, 'rec': tr, 'f1': tf1, 'val_acc': va, 'val_prec': vp, 'val_rec': vr, 'val_f1': vf1})
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.out, index=False)

def build_data_pack(idx, data, flags, fit_scaler=False, sc_s=None, sc_c=None):
    pack = {}
    if flags[0]: pack['inp_cnn'] = data['X_cnn'][idx]
    if flags[1]: pack['inp_patch'] = data['X_patch'][idx]
    if flags[2]: pack['inp_mgaf'] = data['X_mgaf'][idx]
    
    if flags[3]:
        if fit_scaler: sc_s = StandardScaler().fit(data['X_spec'][idx])
        pack['inp_spec'] = sc_s.transform(data['X_spec'][idx]).astype(np.float32)
    if flags[4]:
        if fit_scaler: sc_c = StandardScaler().fit(data['X_color'][idx])
        pack['inp_color'] = sc_c.transform(data['X_color'][idx]).astype(np.float32)
    return pack, sc_s, sc_c

def run_experiment(data, flags, exp_name, epochs=50, bs=16):
    out_dir = os.path.join("logs", exp_name)
    os.makedirs(out_dir, exist_ok=True)
    stats = []

    for t in range(1, 11):
        print(f"\n>>> Running {exp_name} | Trial {t}/10")
        tr_idx, te_idx = train_test_split(np.arange(len(data['y'])), test_size=0.2, stratify=data['y'], random_state=100+t)
        tr_idx, va_idx = train_test_split(tr_idx, test_size=0.25, stratify=data['y'][tr_idx], random_state=1000+t)

        tr_pack, sc_s, sc_c = build_data_pack(tr_idx, data, flags, True)
        va_pack, _, _ = build_data_pack(va_idx, data, flags, False, sc_s, sc_c)
        te_pack, _, _ = build_data_pack(te_idx, data, flags, False, sc_s, sc_c)

        y_tr, y_va = tf.keras.utils.to_categorical(data['y'][tr_idx], 5), tf.keras.utils.to_categorical(data['y'][va_idx], 5)

        model = build_pamif_net(data['cnn_shape'], data['cnn_mode'], data.get('patch_shape'), data.get('mgaf_shape'), 
                                data.get('spec_len'), data.get('X_color').shape[1], flags)
        
        model.compile(optimizer=optimizers.Adam(3e-4), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                      metrics=['accuracy', QWK_Cls(5)])

        trial_dir = os.path.join(out_dir, f"trial_{t}")
        os.makedirs(trial_dir, exist_ok=True)
        cb = [callbacks.ReduceLROnPlateau(monitor='val_qwk', mode='max', factor=0.5, patience=6, verbose=0),
              callbacks.EarlyStopping(monitor='val_qwk', mode='max', patience=12, restore_best_weights=True),
              EpochLogger(tr_pack, data['y'][tr_idx], va_pack, data['y'][va_idx], os.path.join(trial_dir, "epoch_logs.csv"))]

        t0 = time.time()
        model.fit(tr_pack, y_tr, validation_data=(va_pack, y_va), epochs=epochs, batch_size=bs, callbacks=cb, verbose=0)
        train_time = time.time() - t0

        # Testing
        pred = model.predict(te_pack, verbose=0).argmax(axis=1)
        p, r, f1, _ = precision_recall_fscore_support(data['y'][te_idx], pred, average='macro', zero_division=0)
        acc, mae = accuracy_score(data['y'][te_idx], pred), days_mae(data['y'][te_idx], pred)

        with open(os.path.join(trial_dir, "metrics.json"), "w") as f:
            json.dump({'acc': acc, 'precision': p, 'recall': r, 'f1': f1, 'mae': mae, 'sec': train_time}, f, indent=2)
        save_cm_tif(data['y'][te_idx], pred, os.path.join(trial_dir, "cm.tif"))
        stats.append({'Trial': t, 'Acc': acc, 'Prec': p, 'Rec': r, 'F1': f1, 'MAE': mae, 'Sec': train_time})

    df = pd.DataFrame(stats)
    summary = pd.concat([df.mean().add_suffix('_mean'), df.std().add_suffix('_std')]).to_frame().T
    summary.insert(0, 'Combo', exp_name)
    summary.to_csv(os.path.join(out_dir, "summary_10_trials.csv"), index=False)
    print(summary)

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    dataset = load_dataset()
    
    # Define ablation combinations (CNN, PATCH, MGAF, SPEC, COLOR)
    combos = {
        "CNN_only":         (1,0,0,0,0),
        "CNN_PATCH":        (1,1,0,0,0),
        "CNN_MGAF":         (1,0,1,0,0),
        "CNN_SPEC_COLOR":   (1,0,0,1,1),
        "PATCH_MGAF":       (0,1,1,0,0),
        "PAMIF_NET_ALL":    (1,1,1,1,1)
    }

    for name, flg in combos.items():
        run_experiment(dataset, flg, exp_name=name)
        
    print("\n✅ All ablation trials completed successfully. Check the 'logs' directory.")