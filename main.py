%pip install seaborn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('IRIS.csv')

df.head()

df.info()

df.describe()

sns.scatterplot(df,x="sepal_length",y= "sepal_width")
plt.show()

sns.histplot(df,x="sepal_length")
plt.show()

sns.pairplot(df)
plt.show()

X=df.drop("species",axis=1)
y=df["species"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy)

confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confusion)

classification=classification_report(y_test,y_pred)
print("Classification Report:\n",classification)
