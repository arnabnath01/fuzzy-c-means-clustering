# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator


# %%
filename = "housing.csv"
data=pd.read_csv(filename)
# print(data) 
data

# %%
# data = data.iloc[:,  :-1]
# data=data.loc[:,['longitude','latitude','median_income','median_house_value','population']]
data=data.loc[:,['median_income','median_house_value']]
# plt.scatter(data.latitude,data.longitude)
data

# %%
data=np.array(data)
data

# %%

# dec. an empty list to store the within-cluster sum of squares (WCSS)
wcss = []
K = range(1,40)  # Change this range according to your needs


for k in K:
    # Create a KMeans model with 'k' clusters
    kmeanModel = KMeans(n_clusters=k)

    kmeanModel.fit(data)
    
    wcss.append(kmeanModel.inertia_)

# Use the KneeLocator function to find the 'elbow point' in the WCSS curve
kn = KneeLocator(K, wcss, curve='convex', direction='decreasing')

# Print the optimal number of clusters
print('The optimal number of clusters is:', kn.knee)

# Store the optimal number of clusters for later use
optimal_k = kn.knee


plt.figure(figsize=(16,8))
plt.plot(K, wcss, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.show()

# %%
# Initialize an empty list to store the within-cluster sum of squares (WCSS)
wcss = []

# Define the range of 'k' values to test
K = range(1,40)  # Change this range according to your needs

# Loop over the 'k' values
for k in K:
    # Create a KMeans model with 'k' clusters
    kmeanModel = KMeans(n_clusters=k)
    
    # Fit the model to the data
    kmeanModel.fit(data)
    
    # Append the model's WCSS to the list
    wcss.append(kmeanModel.inertia_)

# Use the KneeLocator function to find the 'elbow point' in the WCSS curve
kn = KneeLocator(K, wcss, curve='convex', direction='decreasing')

# Print the optimal number of clusters
print('The optimal number of clusters is:', kn.knee)

# Store the optimal number of clusters for later use
optimal_k = kn.knee

# %%

m=2
U=np.random.rand(data.shape[0],optimal_k)
# U=U/U.sum(axis=1).reshape(-1,1)
U/=U.sum(axis=1)[:,np.newaxis]

# %%
c=np.sum((U[:,0]**m)[:,np.newaxis]*data,axis=0)/np.sum(U[:,0]**m)

# %%
def calc_centroids(data, U, m):    
    centroids = np.zeros((optimal_k, data.shape[1]))
    for i in range(optimal_k):
        centroids[i:] = np.sum((U[:,i]**m)[:,np.newaxis]*data, axis=0) / np.sum(U[:,i]**m)
    return centroids

centroids = calc_centroids(data, U, m)


# %%
def calculate_membership (data, Centroids, optimal_k , m):
    U_new=np.zeros((data.shape[0],optimal_k))
    for i in range (optimal_k):
        # finds the Euclidean distance from each data point to the centroid of the current cluster
        U_new[:,i]=np.linalg.norm(data-Centroids[i,:],axis=1)
    # upates the membership values based on the distance
    U_new=1/ (U_new ** (2/(m-1)) * np.sum((1/U_new) ** (2/(m-1)) , axis=1 )[:,np.newaxis] )
    return U_new

# %%
U_new=np.zeros((data.shape[0],optimal_k))
U_new=calculate_membership(data,centroids,optimal_k,m)

U_new


# %%
labels=np.argmax(U_new,axis=1)
labels

# %%
sns.scatterplot(data=pd.DataFrame(data), x=data[:,0], y=data[:,1], hue=labels, palette='viridis')

# %%
max_itteration=300

for itteration in range(max_itteration):
    cent=calc_centroids(data,U,m)
    U_new=calculate_membership(data,cent,optimal_k,m)
    
    if np.linalg.norm(U_new-U)<0.0001:
        break
    U=U_new
    
    labels=np.argmax(U_new,axis=1)

# %%
sns.scatterplot(data=pd.DataFrame(data), x=data[:,0], y=data[:,1], hue=labels, palette='viridis')


