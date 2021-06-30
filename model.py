'''import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('C:/Users/reach/OneDrive/Desktop/Projects/sales.csv')

dataset['rate'].fillna(0, inplace=True)

dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

X = dataset.iloc[:, :3]

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]

X['rate'] = X['rate'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X, y)

pickle.dump(regressor, open('C:/Users/reach/OneDrive/Desktop/Projects/model.pkl','wb'))

model = pickle.load(open('C:/Users/reach/OneDrive/Desktop/Projects/model.pkl','rb'))
print(model.predict([[4, 300, 500]]))'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df=pd.read_csv("C:/Users/reach/OneDrive/Desktop/Projects/Consumo_cerveja.csv")
df=df.loc[:364,:]
for i in range(0,365):
    df['Temperatura Maxima (C)'][i]=df['Temperatura Maxima (C)'][i].replace(',','.')
    df['Temperatura Media (C)'][i]=df['Temperatura Media (C)'][i].replace(',','.')
    df['Temperatura Minima (C)'][i]=df['Temperatura Minima (C)'][i].replace(',','.')
    df['Precipitacao (mm)'][i]=df['Precipitacao (mm)'][i].replace(',','.')
df['Temperatura Maxima (C)']=df['Temperatura Maxima (C)'].astype('float64')
df['Temperatura Media (C)']=df['Temperatura Media (C)'].astype('float64')
df['Temperatura Minima (C)']=df['Temperatura Minima (C)'].astype('float64')
df['Precipitacao (mm)']=df['Precipitacao (mm)'].astype('float64')
X=df[['Temperatura Media (C)','Temperatura Minima (C)','Temperatura Maxima (C)','Final de Semana','Precipitacao (mm)']]
y=df[['Consumo de cerveja (litros)']]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train,Y_train)
import pickle
pickle.dump(regr, open('C:/Users/reach/OneDrive/Desktop/Projects/model.pkl','wb'))
model = pickle.load(open('C:/Users/reach/OneDrive/Desktop/Projects/model.pkl','rb'))
print(model.predict([[4, 300, 500,10,10]]))
