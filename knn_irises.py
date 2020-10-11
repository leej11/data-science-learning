import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


irises = load_iris(return_X_y=True, as_frame=True)
df = pd.merge(irises[0], irises[1], how='inner', left_index=True,
            right_index=True)

X = df[
    [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)'
    ]
]
Y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

knn = KNeighborsClassifier(5)
knn.fit(X_train, Y_train)
knn.score(X_test, Y_test)