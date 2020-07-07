#import the libraries
%matplotlib inline
import matplotlib.pyplot as plt import seaborn; seaborn.set() import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.mlab as mlab import warnings warnings.filterwarnings('ignore')


#Dataset overview
df=pd.read_csv('AB_NYC_2019.csv')
df.isna().sum()


#Data visualization
#Airbnb in different boroughs of NYC (pie chart)
df1=df.groupby('neighbourhood_group').count().sort_values(by='id',ascending=False) x=list(df1[:5]['id'])
y=[1,0,0,0,0] label=[df1.index[0],df1.index[1],df1.index[2],df1.index[3],df1.index[4]] plt.pie(x,labels=label)
plt.pie(y,radius=0.5,colors='w')
plt.title('Airbnb distribution in five different boroughs') 
plt.show()


#Correlation between different features (heatmap)
air=df.drop(['id','host_id','latitude','longitude','reviews_per_month'],axis=1) 
sns.heatmap(air.corr(), annot=True, linewidths=0.1, cmap='Reds')


#Price (violin plot)
df3=df[df.price < 500] #avoid extreme values
violin=seaborn.violinplot(data=df3, x='neighbourhood_group', y='price') 
violin.set_title('Density and distribution of prices for each neighberhood group') 
violin


#Price distribution (location plot)
price=df[df.price < 500] 
img=plt.imread('New_York_City_.jpg') 
plt.figure(figsize=(10,8)) 
plt.imshow(img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92]) 
ax=plt.gca()
price.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price', ax=ax,
           cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.5, zorder=5) 
plt.legend()
plt.show()


#Room type (location plot)
plt.figure(figsize=(9,8)) 
sns.scatterplot(df.longitude,df.latitude,hue=df.room_type,alpha=0.8) 
img=plt.imread('New_York_City_.jpg') 
plt.imshow(img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92]) 
plt.ioff()


#Availability (box plot)
sns.boxplot(data=df, x='neighbourhood_group',y='availability_365', palette='GnBu',saturation=0.8)
