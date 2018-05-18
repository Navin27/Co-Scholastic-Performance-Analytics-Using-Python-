#%matplotlib inline # needed for iPython
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import manifold
from sklearn.datasets import load_iris
import pandas as pd
from highcharts import Highchart



def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # print "Shape:",covariance.shape
    # print "Covariance:", covariance
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        # sorted(s,key=float,reverse=True)
        # q = [s[0],s[1]]
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        # sorted(covariance,key=float,reverse=True)
        # q = [covariance[0],covariance[0]]
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(3, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
    
        
        
       
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    # storing data x,y and label to csv file
    df=pd.DataFrame(data=X,columns=["PCA1","PCA2"])
    df['labels']=labels
    df.to_csv('gmm.csv')
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)
    #return iris_data
        
        		
def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    labels = kmeans.fit_predict(X)
    #print labels
    #plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
        


        

print ("------------------")
print ("Generate some data")
print ("------------------")
data=pd.read_csv('student_data.csv')
x=data.iloc[:,2:]


pca= PCA(n_components=2,svd_solver='auto')
X = pca.fit_transform(x)
print("==========Label data============")



X = X[:, ::-1] # flip axes for better plotting
print("======After Flipping=========")
print(X)

print ("---------------------------------")
print ("Plot the data with K Means Labels")
print ("---------------------------------")
kmeans = KMeans(1, random_state=0)
labels = kmeans.fit(X).predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
plt.show()

print ("-----------------------------")
print ("How many components required?")
print ("-----------------------------")
n_components =np.arange(1,21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X)
          for n in n_components]

plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');
plt.show()


#print("n_components:",n_components)
ybic_point=[m.bic(X) for m in models]
xaic_point=[m.aic(X) for m in models]

minimum =np.argmin(np.array(ybic_point))
print("minimum_index:",minimum)
print("minimum_value:",np.min(np.array(ybic_point)))
print("components required:",n_components[minimum])
n_clusters = n_components[minimum] 

print ("--------------")
print ("kmeans cluster")
print ("--------------")
kmeans = KMeans(n_clusters, random_state=0)
plot_kmeans(kmeans, X)	
plt.show()
print ("--------------")
print ("GMM Clustering")
print ("--------------")
gmm = GaussianMixture(n_clusters, covariance_type='full',random_state=0).fit(X) # E-M method
labels = gmm.predict(X)
plot_gmm(gmm, X)


csv_data1=pd.read_csv("student_data.csv")
csv_data=pd.read_csv("gmm.csv")
csv_data1["Labels"]=csv_data["labels"]
csv_data1["PCA1"]=csv_data["PCA1"]
csv_data1["PCA2"]=csv_data["PCA2"]

#  Generating the data for plotting the scatter chart using highcharts
df=pd.DataFrame(data=csv_data1)
group=df.Labels.unique();
print(df)
name=[]
result=dict()
for level in group:
    name.append(str(level))
    data=[]
    for index,row in df.iterrows():
        temp=dict()
        if row['Labels']==level:
            temp['x']=row['PCA1']
            temp['y']=row['PCA2']
            temp['name']=row['srid']
            data.append(temp)
    result[level]=data

            
       
# Draw highchart
H = Highchart(width=850, height=400)

options = {
	'chart': {
        'type': 'scatter',
        'zoomType': 'xy'
    },
    'title': {
        'text': 'Co-Scholastic Performance with Different Features'
    },
    'subtitle': {
        'text': 'Class: I Std'
    },
    'xAxis': {
        'title': {
            'enabled': True,
            'text': 'x-PCA'
        },
        'startOnTick': True,
        'endOnTick': True,
        'showLastLabel': True
    },
    'yAxis': {
        'title': {
            'text': 'y-PCA'
        }
    },
    'legend': {
        'layout': 'vertical',
        'align': 'left',
        'verticalAlign': 'top',
        'x': 100,
        'y': 70,
        'floating': True,
        'backgroundColor': "(Highcharts.theme && Highcharts.theme.legendBackgroundColor) || '#FFFFFF'",
        'borderWidth': 1
    },
    'plotOptions': {
        'scatter': {
            'marker': {
                'radius': 5,
                'states': {
                    'hover': {
                        'enabled': True,
                        'lineColor': 'rgb(100,100,100)'
                    }
                }
            },
            'states': {
                'hover': {
                    'marker': {
                        'enabled': False
                    }
                }
            },
            'tooltip': {
                'headerFormat': '<b>{series.name}</b><br>',
#                'pointFormat': '{point.name}:{point.x} , {point.y} '
                'pointFormat': 'Student Srid:{point.name} '
            }
        }
    },
}
H.set_dict_options(options)
for i in range(len(name)):
    H.add_data_set(result[i], 'scatter', name[i])
#H.add_data_set(data2, 'scatter', 'Male', color='rgba(119, 152, 191, .5)')

H.htmlcontent
H.save_file()
