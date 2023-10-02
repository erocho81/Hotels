#!/usr/bin/env python
# coding: utf-8

# In[1]:


#This is the data study for a hotel dataset from the video https://www.youtube.com/watch?v=S2zBHmkRbhY&t=396s:
#https://absentdata.com/data-analysis/where-to-find-data/

#Este es un estudio de datos para el video https://www.youtube.com/watch?v=S2zBHmkRbhY&t=396s:
#https://absentdata.com/data-analysis/where-to-find-data/

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#Importacion de las hojas del fichero excel
#Excel import for each sheet

xls = pd.ExcelFile('D:/Análisis de datos/DATASETS/hotel dataset youtube/hotel_revenue_historical_full-2.xlsx')
df1 = pd.read_excel(xls, '2018')
df2 = pd.read_excel(xls, '2019')
df3 = pd.read_excel(xls, '2020')


# In[3]:


#Revisamos los df
#We check the dfs

#year 2018
df1.head()


# In[5]:


#year 2019
df2.head()


# In[6]:


#year 2020
df3.head()


# In[4]:


#We concat/union the dataframes
hotels= pd.concat([df1, df2, df3])

hotels.head()


# In[4]:


#Let's check if there are Na in some of the columns
#Queremos revisar si hay NA's en alguna de las columnas que utilizaremos
hotels.hotel.isna().any()


# In[12]:


hotels.lead_time.isna().any()


# In[13]:


#we check the columns of the dataset
hotels.columns


# In[14]:


#data types
hotels.dtypes


# In[5]:


#We create a new column with "stays_in_weekend_nights" + "stays_in_week_nights" to get the total nights

hotels["total_nights"]= hotels["stays_in_weekend_nights"] + hotels["stays_in_week_nights"]


# In[5]:


#We check:
hotels.head()


# In[17]:


hotels.dtypes


# In[6]:


#Let's check the empty values:

#Revisamos valores vacíos:

total = hotels.isnull().sum().sort_values(ascending=False)
percent = (hotels.isnull().sum()/hotels.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data['Total'] > 0]


# In[7]:


# Con el código siguiente quitaremos los valores según porcentaje de missing del 15%

#With the following code we will remove values with a missing percentage above 15%.

hotels = hotels[missing_data[missing_data['Percent'] < 0.15].index]
hotels


# In[8]:


#If we check again the empty value we will see that the column "company" is gone

#Si revisamos los valores vacíos veremos que la columna "company" ya no está:

total = hotels.isnull().sum().sort_values(ascending=False)
percent = (hotels.isnull().sum()/hotels.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data[missing_data['Total'] > 0]


# In[28]:


#We can also check the columns in order to see that "company is gone"
sorted(hotels)


# In[9]:


#reservations by year
hotels['arrival_date_year'].value_counts().sort_values()


# In[9]:


#Ahora realizamos las visualizaciones para los valores numéricos del df.
#Here we show the visualizations for the numerical values of the df.

hotels.hist(bins=50, figsize=(30,20))


# In[11]:


#Vamos a revisar las visualizaciones de solo algunas de las columnas:

#We are going to check the visualizations for only some columns:

hotels.hist(bins=50, figsize=(30,20), column=["agent", "reservation_status_date", "arrival_date_week_number","total_nights"])


# In[9]:


#Vamos a visualizar los valores medios de "total_nights", "stays_in_weekend_nights","stays_in_week_nights"
# por "reservation_status_date" para ver el num de noches reservadas avg a lo largo del tiempo.
 
#Let's plot the average "total_nights", "stays_in_weekend_nights","stays_in_week_nights" per "reservation_status_date".

hotels.groupby('reservation_status_date')["total_nights", "stays_in_weekend_nights","stays_in_week_nights" ].mean().plot()


# In[10]:


#En este caso visualizaremos la suma de "total_nights","stays_in_weekend_nights","stays_in_week_nights"
# por "reservation_status_date" para ver el num de noches reservadas totales a lo largo del tiempo.

#Let's plot the sum "total_nights","stays_in_weekend_nights","stays_in_week_nights" per "reservation_status_date".

hotels.groupby('reservation_status_date')["total_nights", "stays_in_weekend_nights","stays_in_week_nights" ].sum().plot()


# In[14]:


#Vamos a visualizar los valores medios de "adr" por "reservation_status_date", para ver el precio medio por noche y año

#Let's plot the average "total_nights" per "reservation_status_date", to get the avg price per night and year


hotels.groupby('reservation_status_date')["adr"].mean().plot()


# In[9]:


#Decribimos algunos valores numéricos seleccionando sus columnas.
#We describe some numeric values, selecting the required columns


hotels[["children","booking_changes", "adr","required_car_parking_spaces", "total_of_special_requests", "previous_cancellations","is_canceled", "babies","adults","stays_in_week_nights",
 "stays_in_weekend_nights","total_nights", "lead_time"]].describe()


# In[10]:


#Análisis bivariante para "total_nights" y "adr".

#This is a bivariant visualization for "total_nights" and "adr".

var = 'total_nights'
data = pd.concat([hotels['adr'], hotels['total_nights']], axis=1)
data.plot.scatter(x='total_nights', y='adr')

#We see that there are some outliers, specially for adr


# In[10]:


#We are going to use z-score to check outliers.

#The Z-Score (also known as the Standard Score) is a statistic that measures how many standard deviations
#a data point is from the mean. A larger Z-score shows that the data point is farther away from the mean.

#This is important because most data points are near the mean in a normally distributed data set. 
#A data point with a large Z-score is farther away from most data points and is likely an outlier.

#When working with normal distributions, data points three standard deviations above the mean are considered outliers.

#This is because 99.7% of the points are within 3 standard deviations of the mean in a normal distribution. 
#This means that all points with a Z-score greater than 3 should be removed.

from scipy import stats
z = stats.zscore(hotels['adr'])
z_abs = np.abs(z)

print(z_abs)



# In[11]:


#we define a threshold to identify outliers

threshold = 3
print(np.where(z > 3))


# In[13]:


#To remove values with an z score >=3 we use:

hotels['adr'] = hotels['adr'].mask(np.abs(stats.zscore(hotels['adr'])) >= 3)


# In[14]:


#We recheck the visualization for "total_nights" and "adr".

var = 'total_nights'
data = pd.concat([hotels['adr'], hotels['total_nights']], axis=1)
data.plot.scatter(x='total_nights', y='adr')

#The visualization is now adjusted below adr 250, the previous value above 5000 adr is gone


# In[48]:


#NO HACE FALTA?

#We will also use the Inter-Quartile Range to detect outliers

#The Inter-Quartile Range (IQR) is the difference between the data’s third quartile and first quartile.
#We define Q1 as the first quartile, which means that 25% of the data lies between the minimum and Q1.
#We define Q3 as the third quartile of the data, meaning that 75% of the data lies between the dataset minimum and Q3.

#We can use the NumPy function percentile() to find Q1 and Q3 and then find the IQR.

#Q1 = np.percentile(hotels['adr'], 25, interpolation = 'midpoint')
#Q3 = np.percentile(hotels['adr'], 75, interpolation = 'midpoint')
#IQR = Q3 - Q1



# In[49]:


#We print the IQR

#print(IQR)


# In[54]:


#We will now define our upper and lower bounds as follows:

#upper=Q3+1.5*IQR
#upper_array=np.array(hotels['adr']>=upper)
#print("Upper Bound:",upper)
#print(upper_array.sum())


# In[55]:


#Below Lower bound

#lower=Q1-1.5*IQR
#lower_array=np.array(hotels['adr']<=lower)
#print("Lower Bound:",lower)
#print(lower_array.sum())


# In[51]:


#we can get the indices for the points which fit the criteria using np.where.

#print(np.where(upper_bound))
#print(np.where(lower_bound))


# In[58]:


#We will use the dataframe.drop function to drop the outlier points.
#For this, we will have to pass a list containing the indices of the outliers to the function. 
# Removing the outliers

#upper_array = np.where(hotels['adr']>=upper)[0]
#lower_array = np.where(hotels['adr']<=lower)[0]


# In[59]:


#ERROR!!

#hotels.drop(index=upper_array, inplace=True)
#hotels.drop(index=lower_array, inplace=True)


# In[13]:


#Cantidad de reservas por customer type
hotels["customer_type"].value_counts().plot(kind='bar')


# In[40]:


#Cantidad de reservas por meals
hotels["meal"].value_counts().plot(kind='bar')


# In[41]:


#Realizamos un análisis multivariante para las variables numéricas. 
#Obviamente hay relaciones entre stay_in_week_nights y stay_in_weekend_nights con total_nights, 
#ya que hemos creados esta última con la suma de las anteriores.

#Además tenemos relación entre is_repeated_guest con previous_bookings_not_cancelled

#We create a multivariant visualization.
#There is an obvious relation between stay_in_week_nights and stay_in_weekend_nights with total_nights

#Multivariante sin normalizar:
corrmat = hotels.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)


# In[42]:


#Podemos agrupar las variables más relacionadas para ver este detalle más claramente.

#We can group the most related columns to see this more clearly.

corrmat = hotels.corr(method='spearman')
cg = sns.clustermap(corrmat, cmap="YlGnBu", linewidths=0.1);
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
cg


# In[15]:


#Bivariante entre meal y adr con boxplot. 
#Vemos valores más altos para HB

#The following is a bivariant boxplot visualization between entre meal and adr with boxplot. 
#There are higher values for HB

var = 'meal'
data = pd.concat([hotels['adr'], hotels[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="adr", data=data)
plt.xticks(rotation=90);


# In[16]:


#We also need to check if there is any na for "adr" to be able to use algorithms

hotels.adr.isna().any()


# In[17]:


#In this case we will replace the Na values by the average value of the column adr
#First we will get the mean value for the column

mean_value=hotels['adr'].mean()


# In[18]:


#Now we fill de na values with the mean
hotels['adr'].fillna(value=mean_value, inplace=True)


# In[19]:


#Now we recheck if there are Na values for the adr column:
hotels.adr.isna().any()


# In[20]:


#We also check the na's for total_nights
hotels.total_nights.isna().any()


# In[19]:


#LINEAR REGRESSION

#Entrenamiento y Test.
#let's study the relation between ADR and total_nights

from sklearn.model_selection import train_test_split

X = hotels.adr.values #This is the meta_score column

Y = hotels.total_nights.values #This is the global_sales column


X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.2, random_state = 0)


# In[20]:


#We use reshape to avoid the following error for lear regression: Expected 2D array, got 1D array instead
X_train= X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


# In[21]:


#Fitting Simple Linear Regression Model to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)


# In[22]:


#The coefficient of determination, denoted as 𝑅², tells you which amount of variation in 𝑦 can be explained
#by the dependence on 𝐱, using the particular regression model. 
#A larger 𝑅² indicates a better fit and means that the model can better explain the variation
#of the output with different inputs.

#The value 𝑅² = 1 corresponds to SSR = 0. That’s the perfect fit, since the values
#of predicted and actual responses fit completely to each other.

#In this case the value is only 0.00584 which is low for the model

r_sq = regressor.score(X_train, Y_train)
print(f"coefficient of determination: {r_sq}")


# In[23]:


print(f"intercept: {regressor.intercept_}")

#The value of 𝑏₀ is approx 2.988
#This illustrates that the model predicts the response 2.988 when X is zero.


# In[24]:


print(f"slope: {regressor.coef_}")

#The value 𝑏₁ = 0.00444371 means that the predicted response rises by 0.00444371 when X is increased by one.


# In[25]:


#Predicted Response

predicted = regressor.predict(X_test)

print(f"predicted response:\n{predicted}")


# In[28]:


#Visualization of training results

plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')


# In[26]:


#Visualization of the test results

plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')


# In[27]:


#We can use this fitted model to calculate the outputs based on new inputs:

x_new = np.arange(5).reshape((-1, 1))
x_new


# In[75]:


#LOGISTIC REGRESSION
#Vamos a crear un subset categorico.

#Let's create a categorical subset.

hotels_categorical = hotels[['hotel', 'arrival_date_month']] 
hotels_categorical


# In[76]:


#Creamos nuevas columnas categóricas binarizadas:

#We create the new binary columns:

import pandas as pd
hotels_categorical = pd.concat([pd.get_dummies(hotels_categorical[col], prefix=col) for col in hotels_categorical], axis=1)


# In[77]:


#We concat some of the numeric columns to the new categorical ones.

#Unimos varias variables numéricas con las columnas categóricas nuevas.

df_categ = pd.concat([hotels[['total_nights', 'adults']], hotels_categorical], axis=1)
df_categ.head()


# In[78]:


df_categ.columns


# In[79]:


#Changing columns names to remove spaces so they are easier to manage.
#Cambiamos los normbres de las columnas para que sean más fáciles de utilizar.

df_categ.columns = df_categ.columns.str.replace(' ', '_')


# In[68]:


df_categ.columns


# In[80]:


#Vamos a crear un nuevo par de test/train
#We are going to create a new test/train pair

from sklearn.model_selection import train_test_split

X2 = df_categ.drop('hotel_City_Hotel', 1)
y2 = df_categ.hotel_City_Hotel

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y2, test_size=0.30, random_state=42)


# In[81]:


y_test_2


# In[82]:


#Aplicamos el modelo
#We apply the model

from sklearn import linear_model, datasets

logreg = linear_model.LogisticRegression(max_iter=600, solver='lbfgs')
model = logreg.fit(X_train_2, y_train_2)
model



# In[83]:


#We check the predicted data
#Revisamos la información predicha

predicted_2 = model.predict(X_test_2)
predicted_2


# In[84]:


#Creamos una matriz de confusión para revisar cuantos datos han sido correctamente clasificados.
#We create a confusion matrix to check how many data have been correctly/incorrectly classified.

#In this case the false data has been correctly classified, but the true data has been incorrectly classified.

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

#predicted_2 = np.round(predicted_2)
matrix2 = confusion_matrix(y_test_2, predicted_2)
sns.heatmap(matrix2, annot=True, fmt="d", cmap='Blues', square=True)
plt.xlabel("predicción")
plt.ylabel("real")
plt


# In[85]:


#Esta es la accuracy que obtenemos del modelo.
#This is the accuracy obtained:

from sklearn.metrics import accuracy_score

accuracy_score(y_test_2, predicted_2)


# In[86]:


#We can check more detailed information about the trained model:

from sklearn.metrics import classification_report

report = classification_report(y_test_2, predicted_2)
print(report)


# In[89]:


# DECISION TREES

# Load libraries
# Cargamos las librerías
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[90]:


#We split the dataset in features and target variable. "total nights" will be the variable that we want to predict.

#Dividimos de nuevo el dataset en features y target. "total nights" será la variable que queremos predecir

feature_cols = ['hotel_City_Hotel', 'hotel_Resort_Hotel',
       'arrival_date_month_April', 'arrival_date_month_August',
       'arrival_date_month_December', 'arrival_date_month_February',
       'arrival_date_month_January', 'arrival_date_month_July',
       'arrival_date_month_June', 'arrival_date_month_March',
       'arrival_date_month_May', 'arrival_date_month_November',
       'arrival_date_month_October', 'arrival_date_month_September']
X3 = df_categ[feature_cols] # Features
y3 = df_categ.total_nights # Target variable


# In[91]:


X3


# In[92]:


# Split dataset into training set and test set
#Creamos de nuevo test/entrenamiento

X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X3, y3, test_size=0.3, random_state=1) # 70% training and 30% test


# In[93]:


# Creation of Decision Tree classifer object
# Creamos el objeto con el clasificador del arbol.

clf = DecisionTreeClassifier()


# In[94]:


# Train Decision Tree Classifer
# Entrenamos el arbol
clf = clf.fit(X_train_3,y_train_3)


# In[95]:


#Predict the response for test dataset
#Predecimos la respuesta para el dataset test

y_pred_3 = clf.predict(X_test_3)


# In[96]:


# Model Accuracy, 29,47%
# La precisión del modelo es del 29,47%

print("Accuracy:",metrics.accuracy_score(y_test_3, y_pred_3))


# In[101]:


#KNN Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[102]:


#Se asignan 2 variables a X y una a y:
#We assign 2 variables to X and one to y:
X4 = df_categ[['hotel_City_Hotel','hotel_Resort_Hotel']].values
y4 = df_categ['arrival_date_month_April'].values


# In[103]:


#realizamos de nuevo la división test/training
#we split the data into test/training again:

X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X4, y4, random_state=0)
scaler = MinMaxScaler()
X_train_4 = scaler.fit_transform(X_train_4)
X_test_4 = scaler.transform(X_test_4)


# In[104]:


#Definimos k como 3 ya que da un poco mejor accuracy
#We have assigned k=3 since it seems to provide a slightly better accuracy:

n_neighbors = 3

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train_4, y_train_4)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train_4, y_train_4)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_4, y_test_4)))


# In[105]:


#Precisión del modelo:

pred = knn.predict(X_test_4)
print(confusion_matrix(y_test_4, pred))
print(classification_report(y_test_4, pred))

# Tenemos una precision del 92%.
# El recall es del 100%, el modelo ha predicho el 100% de las habitaciones que realmente eran 'arrival_date_month_April'
# F1 es 0,96. Cuanto más cercano a 1, mejor modelo tenemos.
# En Support tenemos 32771 que sí eran 'assigned_room_type_A' y 2716 que no eran.


# In[106]:


#Hacemos fit knn de X4, y4, no de test/training
#We fit X4, y4, not training/test

clf2= knn.fit(X4, y4)


# In[108]:


#Con esto podemos obtener una predicción.

#With this we can try to make a prediction.

print(clf2.predict([[1, 2]]))


# In[ ]:




