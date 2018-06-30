import numpy as np

from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


np.random.seed(123)
def getanchors(x, strategy='kmeans', num=15):
    if strategy is 'random':
	idx = np.arange(x.shape[0])
	np.random.shuffle(idx)
        anchor = x[idx[:10], :]
    elif strategy is 'kmeans':
	K = KMeans(n_clusters=10).fit(x)
	anchor = K.cluster_centers_
    return anchor

	
def getTargetImgs(Imgs, labels):
    if labels.ndim == 2:
 	labels = labels.flatten()

    cls= stable_unique(labels)
    Xm = np.empty((cls.shape[0], Imgs.shape[1]))
    for i, ind in enumerate(cls):
	Xm[i, :] = np.mean(np.squeeze(Imgs[np.where(labels == ind), :]), axis=0)

    return Xm

def pairDst(a, b, metric='euclidean'):
    pairDst = cdist(a, b, metric=metric)
    return pairDst

def stable_unique(x):
    _, idx = np.unique(x, return_index=True)
    return x[np.sort(idx)]

def rbf(x, y, topk=50):
    rbfk = cosine_similarity(x.numpy(), y)
#    rbfkSorted = np.argsort(rbfk, axis=1)[:, :-topk]

#    for colIdx in rbfkSorted.T:
# 	rbfk[np.arange(rbfk.shape[0]), colIdx] = 0 

    return rbfk
    
