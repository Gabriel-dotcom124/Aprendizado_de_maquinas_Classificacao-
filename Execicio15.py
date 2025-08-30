import sklearn
import numpy as np
import pandas as pd
import seaborn as sns

penguim = sns.load_dataset("penguins")

with sns.axes_style('whitegrid'):

  grafico = sns.pairplot(data=penguim.drop(['sex', 'island'], axis=1), hue="species", palette="pastel")

with sns.axes_style('whitegrid'):

  grafico = sns.countplot(data=penguim, x='sex', hue="species", palette="pastel")

with sns.axes_style('whitegrid'):

  grafico = sns.countplot(data=penguim, x='island', hue="species", palette="pastel")

print("Valores ausentes antes do tratamento:")
print(penguim.isnull().sum())

numerical_cols = penguim.select_dtypes(include=np.number).columns
for col in numerical_cols:
    if penguim[col].isnull().any():
        mean_val = penguim[col].mean()
        penguim[col] = penguim[col].fillna(mean_val)

categorical_cols = penguim.select_dtypes(include='object').columns
for col in categorical_cols:
    if penguim[col].isnull().any():
        mode_val = penguim[col].mode()[0]
        penguim[col] = penguim[col].fillna(mode_val)

print("\nValores ausentes após o tratamento:")
print(penguim.isnull().sum())


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd

categorical_cols = penguim.select_dtypes(include='object').columns
categorical_cols = categorical_cols.drop('species', errors='ignore')

nominal_cols = ['island', 'sex']
ordinal_cols = [] 
if nominal_cols:
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_nominal = one_hot_encoder.fit_transform(penguim[nominal_cols])
    encoded_nominal_df = pd.DataFrame(encoded_nominal, columns=one_hot_encoder.get_feature_names_out(nominal_cols))
    encoded_nominal_df.columns = [col + "_nom" for col in encoded_nominal_df.columns]
    penguim = pd.concat([penguim, encoded_nominal_df], axis=1)


if ordinal_cols:
    ordinal_encoder = OrdinalEncoder()
    encoded_ordinal = ordinal_encoder.fit_transform(penguim[ordinal_cols])
    encoded_ordinal_df = pd.DataFrame(encoded_ordinal, columns=ordinal_cols)
    encoded_ordinal_df.columns = [col + "_ord" for col in encoded_ordinal_df.columns]
    penguim = pd.concat([penguim, encoded_ordinal_df], axis=1)

display(penguim.head())


original_categorical_cols = penguim.select_dtypes(include='object').columns.tolist()
if 'species' in original_categorical_cols:
    original_categorical_cols.remove('species')

penguim_processed = penguim.drop(columns=original_categorical_cols)

species_column = penguim_processed.pop('species')
penguim_processed.insert(0, 'species', species_column)

display(penguim_processed.head())

from sklearn.model_selection import train_test_split

X = penguim_processed.drop('species', axis=1)
y = penguim_processed['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_)
plt.show()

n_leaves = model.get_n_leaves()
print(f"A árvore de decisão treinada possui {n_leaves} folhas.")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matriz de confusão')
plt.show()

print("Matriz de confusão:")
print(cm)
print("\nInterpretação:")
print("A matriz de confusão mostra as contagens de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos.")
print("Cada linha representa a classe real, e cada coluna representa a classe prevista.")
print("FPor exemplo, o valor na primeira linha e na primeira coluna mostra o número de instâncias da primeira classe que foram corretamente previstas.")

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(f"Precisão do modelo de Árvore de Decisão: {accuracy:.2%}")

penguim = np.array([[38.2, 18.1, 185.0, 3950.0]])

new_data = pd.DataFrame({
    'bill_length_mm': [38.2],
    'bill_depth_mm': [18.1],
    'flipper_length_mm': [185.0],
    'body_mass_g': [3950.0],
    'island': ['Biscoe'],
    'sex': ['Male']
})

nominal_cols = ['island', 'sex']
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

temp_penguim_for_encoding = sns.load_dataset("penguins")

numerical_cols = temp_penguim_for_encoding.select_dtypes(include=np.number).columns
for col in numerical_cols:
    if temp_penguim_for_encoding[col].isnull().any():
        mean_val = temp_penguim_for_encoding[col].mean()
        temp_penguim_for_encoding[col] = temp_penguim_for_encoding[col].fillna(mean_val)

categorical_cols = temp_penguim_for_encoding.select_dtypes(include='object').columns
for col in categorical_cols:
    if temp_penguim_for_encoding[col].isnull().any():
        mode_val = temp_penguim_for_encoding[col].mode()[0]
        temp_penguim_for_encoding[col] = temp_penguim_for_encoding[col].fillna(mode_val)


one_hot_encoder.fit(temp_penguim_for_encoding[nominal_cols])


encoded_new_data = one_hot_encoder.transform(new_data[nominal_cols])
encoded_new_data_df = pd.DataFrame(encoded_new_data, columns=one_hot_encoder.get_feature_names_out(nominal_cols))
encoded_new_data_df.columns = [col + "_nom" for col in encoded_new_data_df.columns]

new_data_processed = new_data.drop(columns=nominal_cols)
new_data_processed = pd.concat([new_data_processed, encoded_new_data_df], axis=1)

missing_cols = set(X.columns) - set(new_data_processed.columns)
for c in missing_cols:
    new_data_processed[c] = 0

new_data_processed = new_data_processed[X.columns]

predicted_species = model.predict(new_data_processed)

print(f"A espécie do penguim com as características fornecidas é: {predicted_species[0]}")


