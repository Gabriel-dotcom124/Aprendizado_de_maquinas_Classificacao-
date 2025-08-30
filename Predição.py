1. Predição

    #Exemplo: Flor do gênero íris com sépala de 6.39cm e 2.71cm e pétala de 6.03cm e 2.23cm, sendo a primeira medida o comprimento e a segunda a largura, respectivamente.


flor = np.array([[6.39, 2.71, 6.03, 2.23]])

especie = model.predict(flor.reshape(1, -1))
print(especie)

model.classes_

graph