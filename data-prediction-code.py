#Data prediction with Decision Tree

#Prepare the data

df=pd.read_csv('AB_NYC_2019.csv') 
a=df['neighbourhood'].drop_duplicates() 
a=list(a)

df = df[['neighbourhood_group','neighbourhood','room_type', 'price', 'minimum_nights', 'availability_365', 'number_of_reviews']]
for x in range(len(df)):
  value = df['number_of_reviews'].iloc[x] 
  if value <= 24:
    df['number_of_reviews'].iloc[x]=0 
  if value>24 and value <=250:
    df['number_of_reviews'].iloc[x]=1 
  if value>250:
    df['number_of_reviews'].iloc[x]=2 
df = pd.get_dummies(df)

y = df['number_of_reviews']
X = df.drop(['number_of_reviews'], axis = 1)

from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 0)


#Parameter tuning

from sklearn import tree                
result_1 = [0 for i in range(30)]
t1 = [0 for i in range(30)]
maxi1 = 0
a1 = 0
for max_depth in range(1,31,1):
    clf = tree.DecisionTreeClassifier(max_depth = max_depth)
    clf = clf.fit(xTrain, yTrain)
    result_1[max_depth-1] = clf.score(xTest,yTest)
    t1[max_depth-1] = clf.score(xTrain,yTrain)
    if result_1[max_depth-1] > maxi1:
        a1 = max_depth
        maxi1 = result_1[max_depth-1]
plt.plot([i for i in range(1,31,1)], t1, label = "training accuracy")
plt.plot([i for i in range(1,31,1)], result_1, label = "valdiation accuracy")
plt.legend()
plt.show()
print(a1, maxi1)


#Model

from sklearn import tree      
clf = tree.DecisionTreeClassifier(max_depth = a1) 
clf = clf.fit(xTrain, yTrain)

#visualization of decision tree
clf.classes_     
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=xTrain.columns,
                                class_names=['0', '1', '2'],filled=True, 
                                rounded=True,special_characters=True)
graph = graphviz.Source(dot_data) 
graph.format='png'
graph.render()

from IPython.display import display, Image   #display the visualization
Image(graph.render('Source.gv.png'))


#Making predictions

p=int(input('Enter the price of your Airbnb:')) 
mn=int(input('Enter the minimum night of your Airbnb:'))
av=int(input('Enter the availability of your Airbnb::'))
ng=input('Enter the borough of your Airbnb:')
nh=input('Enter the neighbourhood of your Airbnb:')
rt=input('Enter the room type(Entire home/apt; Private room; Shared room):')
neigh= ['Bronx','Brooklyn','Manhattan','Queens','Staten Island']
room=['Entire home/apt','Private room','Shared room']
t1=[0,0,0,0]
t2=[0,0]
t3=[0 for i in range(len(a)-1)]
for i in range(5):
    if ng==neigh[i]:
        t1.insert(i,1)
for n in range(3):
    if rt==room[n]:
        t2.insert(n,1)
for m in range(len(a)):
    if nh==a[m]:
        t3.insert(m,1)
x=[p,mn,av]
x.extend(t1)
x.extend(t2)
x.extend(t3)
x=np.array([x])

res=clf.predict(x)
if res==0:
    print('It seems that your Airbnb is not that popular, try adding some special things to make it outstanding!')
if res==1:
    print('Congrats! Your Airbnb will be very popular - more popular than 75% of the Airbnbs in NYC!')
if res==2:
    print('Congrats! Your Airbnb will be one of the most popular places in town - get ready to be rich!')
    
    
    
    
#Data prediction with KNN

#Prepare data

df=pd.read_csv('AB_NYC_2019.csv')     
df = df[['neighbourhood',  'room_type',  'price',  
         'availability_365', 'number_of_reviews']]
for x in df['number_of_reviews'].index.values:
    value = df['number_of_reviews'].loc[x]
    if value < 24:
        df.set_value(x, 'number_of_reviews', 0)
    if value >= 24 and value < 250:
        df.set_value(x, 'number_of_reviews', 1)
    if value >= 250 and value <= 629:
        df.set_value(x, 'number_of_reviews', 2)
        
y = df['number_of_reviews']
X =  df[['price',  'availability_365']]

from sklearn.model_selection import train_test_split 
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 0)


#Parameter tuning

from sklearn.neighbors import KNeighborsClassifier  
result = []
result2 = []
t = []
for n_neighbors in range(1,101,2):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(xTrain, yTrain)
    test = neigh.score(xTest,yTest) 
    train = neigh.score(xTrain, yTrain) 
    result.append(test)
    result2.append(train)
    t.append(n_neighbors)
print(max(result),t[result.index(max(result))])

plt.plot(t, result,label = "validation accuracy") #visualization of the tuning result
plt.plot(t, result2, label = "training accuracy")
plt.legend()
plt.show()

#Model

idx=np.random.choice(np.arange(len(xTrain)),500,replace=False) 
x_sample=np.array(xTrain.iloc[idx])  
y_sample=np.array(yTrain.iloc[idx])

clf = KNeighborsClassifier(n_neighbors=t[result.index(max(result))], 
                           weights='distance') 
clf.fit(xTrain, yTrain)
from mlxtend.plotting import plot_decision_regions #visualization
plot_decision_regions(x_sample,y_sample,clf=clf)
plt.xlabel('Price')
plt.ylabel('Availability')
plt.xlim(0,500)
plt.ylim(0,365)
plt.title('Using KNN with price and availability to predict popularity (reviews)')
plt.show()

#Making predictions

p = input('Enter the price of your Airbnb: ')  
a = input('Enter the availability of your Airbnb: ')
reviews = clf.predict([[p,a]])
print('Prediction: '),
if reviews == 0:
    print('It seems that your Airbnb is not that popular, try adding some special things to make it outstanding!')
elif reviews == 1:
    print('Congrats! Your Airbnb will be very popular - more popular than 75% of the Airbnbs in NYC!')
else:
    print('Congrats! Your Airbnb will be one of the most popular places in town - get ready to be rich!')
