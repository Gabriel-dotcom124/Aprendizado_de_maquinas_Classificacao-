1. Algoritimo


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=123)
model.fit(predictors_train, target_train)

model.__dict__


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=123)
model.fit(predictors_train, target_train)

data.head(1)

features = np.array([[5.1, 3.5, 1.4, 0.2]])
model.predict(features)

