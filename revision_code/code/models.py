import os, numpy as np, pandas as pd, itertools, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from pipe import *

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def make_seq(df, targets, align='prev', L=3, feats=FEAT):
    """targets: list of (district, year). Returns X (n,L,f), y_res, meta df."""
    X, y, meta = [], [], []
    for d, t in targets:
        g = df[df.District == d].set_index('Year')
        yrs = list(range(t - L, t)) if align == 'prev' else list(range(t - L + 1, t + 1))
        if min(yrs) < 2002: continue
        X.append(g.loc[yrs, feats].values); y.append(g.loc[t, 'res'])
        meta.append((d, t, g.loc[t, 'y'], g.loc[t, 'trend']))
    return np.array(X, float), np.array(y, float), pd.DataFrame(meta, columns=['District', 'Year', 'obs', 'trend'])

def build_tcn(L, F):
    inp = layers.Input((L, F)); x = inp
    for d in [1, 2]:
        r = x
        for _ in range(2):
            x = layers.Conv1D(16, 2, padding='causal', dilation_rate=d)(x)
            x = layers.LayerNormalization()(x); x = layers.Activation('relu')(x); x = layers.Dropout(0.3)(x)
        if r.shape[-1] != 16: r = layers.Conv1D(16, 1, padding='same')(r)
        x = layers.Add()([x, r])
    x = layers.GlobalAveragePooling1D()(x)
    m = Model(inp, layers.Dense(1)(x)); m.compile(Adam(1e-3), 'mse'); return m

def build_lstm(L, F):
    m = Sequential([layers.Input((L, F)),
                    layers.LSTM(16, return_sequences=True, recurrent_dropout=0.2), layers.Dropout(0.4),
                    layers.LSTM(8, recurrent_dropout=0.2), layers.Dropout(0.4),
                    layers.Dense(8, activation='relu'), layers.Dropout(0.4), layers.Dense(1)])
    m.compile(Adam(1e-3), 'mse'); return m

def fit_nn(kind, Xtr, ytr, seed, Xval=None, yval=None, epochs=None):
    tf.keras.utils.set_random_seed(seed)
    xs = MinMaxScaler().fit(Xtr.reshape(-1, Xtr.shape[2])); ys = MinMaxScaler().fit(ytr.reshape(-1, 1))
    sx = lambda X: xs.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    m = (build_tcn if kind == 'tcn' else build_lstm)(Xtr.shape[1], Xtr.shape[2])
    ytr_s = ys.transform(ytr.reshape(-1, 1)).ravel()
    if Xval is not None:
        cb = [EarlyStopping('val_loss', patience=60, restore_best_weights=True),
              ReduceLROnPlateau('val_loss', factor=0.5, patience=10, min_lr=1e-6)]
        h = m.fit(sx(Xtr), ytr_s, epochs=500, batch_size=8, verbose=0, shuffle=True,
                  validation_data=(sx(Xval), ys.transform(yval.reshape(-1, 1)).ravel()), callbacks=cb)
        best = int(np.argmin(h.history['val_loss'])) + 1
    else:
        cb = [ReduceLROnPlateau('loss', factor=0.5, patience=10, min_lr=1e-6)]
        m.fit(sx(Xtr), ytr_s, epochs=max(epochs, 5), batch_size=8, verbose=0, shuffle=True, callbacks=cb)
        best = epochs
    pred = lambda X: ys.inverse_transform(m.predict(sx(X), verbose=0).reshape(-1, 1)).ravel()
    return pred, best, m, (xs, ys)

XGB_GRID = dict(n_estimators=[50, 100, 150], max_depth=[2, 3], learning_rate=[0.01, 0.05, 0.1],
                reg_alpha=[0.1, 0.5, 1.0], reg_lambda=[1.0, 3.0, 5.0], min_child_weight=[3, 5, 7])
XCOLS = FEAT + ['dcode']

def xgb_model(p, seed=42):
    return XGBRegressor(subsample=0.8, colsample_bytree=0.8, gamma=0.5, random_state=seed, verbosity=0, n_jobs=1, **p)

def tune_xgb(df, train_end, folds=5):
    """expanding time-ordered CV: validation years train_end-folds+1 .. train_end."""
    best, bp = 1e18, None
    keys = list(XGB_GRID); combos = list(itertools.product(*[XGB_GRID[k] for k in keys]))
    tr = df[(df.Year >= 2002) & (df.Year <= train_end)]
    for c in combos:
        p = dict(zip(keys, c)); se = 0
        for vy in range(train_end - folds + 1, train_end + 1):
            a = tr[tr.Year < vy]; b = tr[tr.Year == vy]
            m = xgb_model(p).fit(a[XCOLS], a.res); se += ((m.predict(b[XCOLS]) - b.res) ** 2).sum()
        if se < best: best, bp = se, p
    return bp
