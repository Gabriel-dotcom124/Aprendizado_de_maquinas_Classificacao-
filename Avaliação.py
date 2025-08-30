1. Avaliação

  *Posição predita

target_predicted = model.predict(predictors_test)

target_predicted[0:5]

target_predicted.shape


    *Posição Teste


target_test[0:5]

target_test.shape



2. Matriz de confusão

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


confusion_matrix = confusion_matrix(target_test, target_predicted)
print(confusion_matrix)

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, predictors_test, target_test)
plt.show()


3. Acuracia

total = confusion_matrix.sum()
print(total)

acertos = np.diag(confusion_matrix).sum()
print(acertos)

acuracia = acertos / total
print(acuracia)

print(f"{round(100 * acuracia, 2)}%")

from sklearn.metrics import accuracy_score

acuracia = accuracy_score(target_test, target_predicted)
print(acuracia)

print(f"{round(100 * acuracia, 2)}%")


