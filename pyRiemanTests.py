
import numpy as np
from pyriemann.estimation import XdawnCovariances
from pyriemann.embedding import Embedding
import mne
from mne import io
import gen_tools
import data_loading
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns


tmin, tmax = -0., 4
raw = data_loading.get_single_mi('001', 2)
raw = gen_tools.preprocess_highpass(raw, 2, 'iir')
event_id = dict(T1=2, T2=3)

X, y, epochs = gen_tools.epoch_data(raw, tmin, tmax, eeg_reject_uV=1000, scale=None)

nfilter = 4
xdwn = XdawnCovariances(estimator='scm', nfilter=nfilter)
split = train_test_split(X, y, train_size=0.25, random_state=42)
Xtrain, Xtest, ytrain, ytest = split
covs = xdwn.fit(Xtrain, ytrain).transform(Xtest)

lapl = Embedding(metric='riemann', n_components=2)
embd = lapl.fit_transform(covs)

fig, ax = plt.subplots(figsize=(7, 8), facecolor='white')

for cond, label in event_id.items():
    idx = (ytest == label)
    ax.scatter(embd[idx, 0], embd[idx, 1], s=36, label=cond)

ax.set_xlabel(r'$\varphi_1$', fontsize=16)
ax.set_ylabel(r'$\varphi_2$', fontsize=16)
ax.set_title('Spectral embedding of ERP recordings', fontsize=16)
ax.set_xticks([-1.0, -0.5, 0.0, +0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0.0, +0.5, 1.0])
ax.grid(False)
ax.legend()
plt.show()