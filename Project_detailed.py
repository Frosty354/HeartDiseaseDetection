import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt


from sklearn.svm import SVC
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

dataset = pd.read_csv('Heart.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LogisticRegression
lj = LogisticRegression(solver="liblinear").fit(X_train,y_train)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train,y_train)
y_predgnb=gnb.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
knnc = KNeighborsClassifier().fit(X_train,y_train)

from sklearn.tree import DecisionTreeClassifier
cartc = DecisionTreeClassifier(random_state=0).fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=0,verbose=False).fit(X_train,y_train)
y_predrf=rfc.predict(X_test)

from sklearn.ensemble import GradientBoostingClassifier
gbmc = GradientBoostingClassifier(verbose=False).fit(X_train,y_train)

from xgboost import XGBClassifier
xgbc = XGBClassifier().fit(X_train,y_train)

from lightgbm import LGBMClassifier
lgbmc = LGBMClassifier().fit(X_train,y_train)

from catboost import CatBoostClassifier
catbc = CatBoostClassifier(verbose=False).fit(X_train,y_train)

from sklearn.metrics import accuracy_score as accs
modelsc = [lj,gnb,knnc,cartc,rfc,gbmc,xgbc,lgbmc,catbc]

for model in modelsc:
    name = model.__class__.__name__
    predict = model.predict(X_test)
    R2CV = cross_val_score(model,X_test,y_test,cv=10,verbose=False).mean()
    error = -cross_val_score(model,X_test,y_test,cv=10,scoring="neg_mean_squared_error",verbose=False).mean()
    print(name + ": ")
    print("-" * 10)
    print(accs(y_test,predict))
    print(R2CV)
    print(np.sqrt(error))
    print("-" * 30)
    
    
r = pd.DataFrame(columns=["MODELS","R2CV"])
for model in modelsc:
    name = model.__class__.__name__
    R2CV = cross_val_score(model,X_test,y_test,cv=10,verbose=False).mean()
    result = pd.DataFrame([[name,R2CV*100]],columns=["MODELS","R2CV"])
    r = r.append(result)
    
figure = plt.figure(figsize=(20,8))   
sns.barplot(x="R2CV",y="MODELS",data=r,color="k")
plt.xlabel("R2CV")
plt.ylabel("MODELS")
plt.xlim(0,100)
plt.title("MODEL ACCURACY COMPARISON")
plt.show()

from sklearn.metrics import confusion_matrix
cm1=confusion_matrix(y_test, y_predgnb)
cm2=confusion_matrix(y_test, y_predrf) 

#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components= None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

per_var=np.round(pca.explained_variance_ratio_ *100,decimals = 1)
labels= ['PC' + str(i)for i in range (1,len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1),height= per_var,tick_label= labels )
plt.ylabel("% of Explained Variance")
plt.xlabel("Principal Components")
plt.title("Scree Plot")
plt.show()  