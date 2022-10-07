import pandas as pd
import os
# Changing the working directory to the specified path--
os.chdir(r"D:\vina\Skripsi\PCA")
df = pd.read_csv("Bismillah PCA2.csv") # dataset
print( len(df))
print( df.head())

from sklearn.preprocessing import StandardScaler
features = ['TGS 2600', 'TGS 2602', 'TGS 2611', 'TGS 2620', 'TGS 2612', 'TGS 826']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['Target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
                          , columns = ['PC1', 'PC2', 'PC3', 'PC4'])
print( len(principalDf))
print( principalDf.head())

finalDf = pd.concat([principalDf, df[['Target']]], axis = 1)
print( len(finalDf))
print( finalDf.head())

import matplotlib.pyplot as plt
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC1', fontsize = 15)
ax.set_ylabel('PC2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Berformalin', 'Tanpa Formalin']
colors = ['r', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Target']==target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
              , finalDf.loc[indicesToKeep, 'PC2']
              , c = color
              , s = 15)
ax.legend(targets)
ax.grid()

pca.explained_variance_ratio_

import numpy as np
(np.round(pca.explained_variance_ratio_,decimals=3)*100)

