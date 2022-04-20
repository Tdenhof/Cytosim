import csv
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import KMeans
import scipy 
import numpy as np
import scipy.cluster.hierarchy as sch # https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import argparse
import os 
from sklearn.neighbors import NearestCentroid
parser = argparse.ArgumentParser()
parser.add_argument("--file",type = str)
parser.add_argument("--expName",type = str, default = "alignmentOut")
parser.add_argument("--expPath",type = str)
parser.add_argument("--frameNumber", type = int, default = 400)
parser.add_argument("--nClusters",type = int, default = 4)
opt = parser.parse_args()
expPath = opt.expPath + '/' + opt.expName
FILE = opt.file
data = pd.read_csv(FILE)

#Make output directory location if needed
exists = os.path.exists(expPath)
if not exists:
  os.mkdir(expPath)

##METHODS##
def preprocessFrames(data):
  frames = []
  subframes = []
  frameN = 1
  for i in range(len(data)):
    item = data.iloc[i][0]
    if item.startswith('% frame'):
      frameN = frameN + 1
    if (item.startswith('%')) == False:
      subframes.append([item.split(),frameN])
  df = pd.DataFrame(subframes)

  return df
def preprocess(data):
  df = preprocessFrames(data)
  nf = max(df[1])
  data = []
  for i in range(nf):
    temp = df[df[1] == i][0]
    temp.reset_index()
    for item in temp:
      frame = i
      id = item[1]
      x = float(item[3])
      y = float(item[4])
      dirx = float(item[5])
      diry = float(item[6])
      cosinus = float(item[7])
      rho,phi = cart2pol(dirx,diry)
      data.append([frame,id,x,y,dirx,diry,cosinus,rho,phi])
  columns = ['frame','id','x','y','dirx','diry','cs','rho','phi']
  return pd.DataFrame(data,columns = columns)
    
## Cartesian to polar 
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

#Polar to Cartesian 
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
#Polar to Vector 
def polarvec(data):
  rhos = []
  phis = []
  for i in range(len(data)):
    x = data.iloc[i].dirx
    y = data.iloc[i].diry
    rho, phi = cart2pol(x,y)
    rhos.append(rho)
    phis.append(phi)
  data['rho'] = rhos 
  data['phi'] = phis 
  return data

def clusterRiver2(df,frame,ncluster):
  data = clusterProcessingRiver(df,frame)
  d = pd.DataFrame(data.phi)
  cluster = AgglomerativeClustering(n_clusters = ncluster, affinity='euclidean', linkage ='ward') # we can set a distance_threshold so only fibers within a certain distance get clustered 
  out = cluster.fit_predict(d)
  plt.scatter(data.x, data.y, c=cluster.labels_, cmap='rainbow')
  plt.legend(["phi"], ncol = data.phi, loc = "upper left")
  return out,cluster

def clusterProcessingRiver(df,frame):
  data = []
  fData = df[df.frame == frame]
  fData.reset_index
  for i in range(len(fData)):
    item = fData.iloc[i]
    data.append([item[2],item[3],item[8]])
  df = pd.DataFrame(data,columns = ['x','y','phi'])
  return df

def outlierProcessing(df):
  df = df.reset_index()
  xy = []
  for i in range(len(df)):
    xy.append([df.iloc[i].x,df.iloc[i].y])
  return pd.DataFrame(xy, columns = ['x','y'])

def outlierDetection(df):
  xy = outlierProcessing(df)
  data = xy.values
  lof = LocalOutlierFactor()
  yhat = lof.fit_predict(data)
  return yhat

## NEED TO EDIT 
def allClusterOutliers(frame,numClust):
  final = []
  for i in range(numClust):
    data = frame[frame.cluster == i]
    outliers = outlierDetection(data)
    data['outliers'] = outliers
    final.append(data)
  return pd.concat(final)
def calcCentroid(df):
    out = []
    for i in range(opt.nClusters):
        cluster = df[df.cluster == i]
        x0 = cluster.x
        y0 = cluster.y
        centroid = (sum(x0) / len(x0), sum(y0) / len(y0))
        c0 = []
        for i in range(len(x0)):
            c0.append(((x0.iloc[i] - centroid[0])**2 + (y0.iloc[i] - centroid[1])**2)**0.5)
        out.append(c0)
    dscdfs = []
    for i in range(opt.nClusters):
      dscdfs.append(describeList(out[i]))
    fout = pd.concat(dscdfs,axis = 1)
    fout.columns = range(opt.nClusters)
    return fout
def describeList(lst):
  lst = pd.DataFrame(lst)
  return lst.describe()
## EXECUTION ##
# Preprocess Data
df = preprocess(data)
#cluster data 
clusterRiver2(df,opt.frameNumber,opt.nClusters)
data = clusterProcessingRiver(df,opt.frameNumber)
d = pd.DataFrame(data.phi)
cluster = AgglomerativeClustering(n_clusters = opt.nClusters, affinity='euclidean', linkage ='ward')
out = cluster.fit_predict(d)
frame = df[df.frame == opt.frameNumber]
frame['cluster'] = out
frame = allClusterOutliers(frame, opt.nClusters)
outlierFree = frame[frame.outliers != -1]

# OUTPUTS
#Outlier Free Data 
outlierFree.to_csv(expPath + '/outlierFree.csv')
#values -- IQR
q3, q1 = np.percentile(outlierFree.phi, [75, 25])
pd.DataFrame([[q1,q3,q3-q1]],columns = ['q1','q3','IQR']).to_csv(expPath + '/IQRPhiVals.csv')
#df of centroids based on clusters
centroids = calcCentroid(outlierFree)
centroids.to_csv(expPath + '/centroids.csv')