import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import gaussian_kde

'''S008 mean and standard deviation to normalize GPS inputs'''
s008_mean=[757.8,555.4,2.1]
s008_SD=[4.5,68.0,0.9]


def load_dataset(filename):
    '''
    :param filename of the dataset to load and LIDAR format parameteres
    :return: [GPS,LIDAR,labels,LOS/NLOS] dataset with LOS available only for s008 and s009
    '''
    npzfile = np.load(filename)
    POS=npzfile['POS']
    for i in range(0,3):
        POS[:,i]=(POS[:,i]-s008_mean[i])/s008_SD[i]
    Y=npzfile['Y']
    return POS,Y



POS_tr, Y_tr = load_dataset('./data/s008.npz')
POS_val, Y_val,=load_dataset('./data/s009.npz')
Y_tr=np.argmax(Y_tr,axis=1)
Y_val=np.argmax(Y_val,axis=1)
test_scores=[]
centers=range(1,20)
for k in centers:
    neigh = KNeighborsClassifier(n_neighbors= k,weights='distance')
    neigh.fit(POS_tr,Y_tr)
    test_scores.append(neigh.score(POS_val,Y_val))
plt.scatter(centers,test_scores,label='val')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
best_neigh = KNeighborsClassifier(n_neighbors= np.argmax(test_scores),weights='distance')
preds=neigh.predict(POS_val)
errors=POS_val[np.where((preds!=Y_val)*1==1)]
plt.hist2d(errors[:,0], errors[:,1], (50, 50), cmap=plt.cm.jet)
plt.colorbar()
plt.show()


