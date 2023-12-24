'''import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from tensorflow import keras

# Carregar dados do CSV
dados = pd.read_csv('IC.csv', encoding='latin-1', delimiter=';', decimal=',')

# Separar dados de entrada (X) e saída (y)
X = dados[['Brix', 'DO', 'pH']].values
y = dados[['substrato', 'produto', 'células']].values

# Definir as configurações de camadas e neurônios a serem testados
camadas_ocultas = [1, 2, 3]  # Número de camadas ocultas
neuronios = [5, 10, 15]  # Número de neurônios em cada camada oculta

# Inicializar variáveis para armazenar a melhor arquitetura encontrada
melhor_arquitetura = None
melhor_erro = np.inf

# Testar todas as combinações de camadas e neurônios
for num_camadas in camadas_ocultas:
    for num_neuronios in neuronios:
        # Definir a arquitetura da rede neural
        model = keras.Sequential()
        model.add(keras.layers.Dense(num_neuronios, activation='relu', input_shape=(3,)))
        for _ in range(num_camadas):
            model.add(keras.layers.Dense(num_neuronios, activation='relu'))
        model.add(keras.layers.Dense(3))

        # Inicialização dos pesos da rede
        initial_weights = model.get_weights()

        # Função de erro para otimização com Levenberg-Marquardt
        def error_function(params):
            model.set_weights(params)
            predictions = model.predict(X)
            return np.ravel(predictions - y)

        # Otimização com Levenberg-Marquardt
        result = least_squares(error_function, initial_weights.flatten())

        # Calcular o erro para a arquitetura atual
        error = np.sum(result.fun ** 2)

        # Verificar se a arquitetura atual é a melhor encontrada até agora
        if error < melhor_erro:
            melhor_arquitetura = model
            melhor_erro = error

# Imprimir a melhor arquitetura encontrada
print("Melhor arquitetura:")
print(melhor_arquitetura.summary())
print("Erro mínimo encontrado:", melhor_erro)
'''






import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Carrega os dados do arquivo CSV
dados = pd.read_csv('IC.csv', encoding='latin-1', delimiter=';', decimal=',')

# Separa os dados de entrada e saída
entradas = dados[['Brix', 'DO', 'pH']].values
saidas = dados[['substrato', 'produto', 'células']].values

# Separa 50 pontos para simulação e os outros 150 para treinamento, teste e validação
entradas_treinamento, entradas_outros, saidas_treinamento, saidas_outros = train_test_split(
    entradas, saidas, test_size=0.7, random_state=42)
entradas_teste, entradas_validacao, saidas_teste, saidas_validacao = train_test_split(
    entradas_outros, saidas_outros, test_size=0.5, random_state=42)

def criar_rede_neural(num_camadas_ocultas, neuronios_por_camada, input_dim, output_dim, dropout_rate):
    model = tf.keras.Sequential()
    
    # Adiciona a primeira camada oculta
    model.add(tf.keras.layers.Dense(neuronios_por_camada[0], activation='tanh', input_shape=(input_dim,)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Adiciona as camadas ocultas intermediárias
    for i in range(1, num_camadas_ocultas):
        model.add(tf.keras.layers.Dense(neuronios_por_camada[i], activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Adiciona a camada de saída
    model.add(tf.keras.layers.Dense(output_dim, activation='relu'))
    
    return model

# Especificação da rede neural
num_camadas_ocultas = 3
neuronios_por_camada = [32,16,32]
input_dim = 3
output_dim = 3
dropout_rate = 0.01 # prevenção de overfitting

# Cria a rede neural
modelo = criar_rede_neural(num_camadas_ocultas, neuronios_por_camada, input_dim, output_dim, dropout_rate)

# Compila o modelo
modelo.compile(optimizer='adam', loss='mean_squared_error')

# Treina o modelo
historico = modelo.fit(entradas_treinamento, saidas_treinamento, epochs=500, batch_size=10, validation_data=(entradas_validacao, saidas_validacao))

# Avalia o modelo
resultado = modelo.evaluate(entradas_teste, saidas_teste)
print('Perda (Loss) do modelo:', resultado)

# Gera gráfico da perda durante o treinamento
plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend(['Treinamento', 'Validação'])
plt.show()

# Gera a figura da rede neural
tf.keras.utils.plot_model(modelo, to_file='rede_neural.png', show_shapes=True, show_layer_names=True)

# Comparação das saídas
saidas_preditas = modelo.predict(entradas_teste)
plt.figure(figsize=(10, 5))
plt.plot(saidas_teste[:, 0], label='Saída Real (substrato)')
plt.plot(saidas_teste[:, 1], label='Saída Real (produto)')
plt.plot(saidas_teste[:, 2], label='Saída Real (células)')
plt.plot(saidas_preditas[:, 0], label='Saída Predita (substrato)')
plt.plot(saidas_preditas[:, 1], label='Saída Predita (produto)')
plt.plot(saidas_preditas[:, 2], label='Saída Predita (células)')
plt.title('Comparação das Saídas')
plt.xlabel('Amostra')
plt.ylabel('Valor')
plt.legend()
plt.show()

# Comparação das saídas
saidas_preditas = modelo.predict(entradas_teste)
num_saidas = saidas_teste.shape[1]
nomes_saidas = dados.columns[-num_saidas:]  # Lista de nomes de saída do conjunto de dados

fig, axs = plt.subplots(num_saidas, 1, figsize=(10, 5*num_saidas), sharex=True)

for i in range(num_saidas):
    axs[i].plot(saidas_teste[:, i], label='Saída Real')
    axs[i].plot(saidas_preditas[:, i], label='Saída Predita')
    axs[i].set_title(f'Comparação - {nomes_saidas[i]}')
    axs[i].set_xlabel('Amostra')
    axs[i].set_ylabel('Valor')
    axs[i].legend()

plt.tight_layout()
plt.show()

import graphviz

def criar_fluxograma(modelo):
    dot = graphviz.Digraph()
    
    # Adiciona os nós correspondentes às camadas
    for i, layer in enumerate(modelo.layers):
        dot.node(str(i), str(layer.__class__.__name__))
    
    # Adiciona as conexões entre os nós das camadas
    for i, layer in enumerate(modelo.layers):
        if i < len(modelo.layers) - 1:
            dot.edge(str(i), str(i + 1))
    
    return dot

# Cria a rede neural
modelo = criar_rede_neural(num_camadas_ocultas, neuronios_por_camada, input_dim, output_dim, dropout_rate)

# Cria o fluxograma da rede neural
fluxograma = criar_fluxograma(modelo)

# Salva o fluxograma em um arquivo
fluxograma.render('fluxograma_rede_neural', format='png', cleanup=True)







'''
# RODADA DE TESTES ATIVADAAAAAAAAAAA!

# Lista de funções de ativação a serem testadas
ativacoes = ['relu', 'sigmoid', 'tanh']

# Lista de números de neurônios na camada oculta a serem testados
neuronios_oculta = [8, 16, 32]

melhor_perda = float('inf')  # Valor inicial da melhor perda
melhor_num_camadas = 0  # Valor inicial do melhor número de camadas ocultas
melhor_ativacao_oculta = ''  # Valor inicial da melhor função de ativação das camadas ocultas
melhor_ativacao_entrada = ''  # Valor inicial da melhor função de ativação da camada de entrada
melhor_neuronios_por_camada = []  # Valor inicial dos melhores números de neurônios por camada

# Loop para testar todas as combinações de número de camadas ocultas, função de ativação das camadas ocultas e função de ativação da camada de entrada
for num_camadas, ativacao_oculta, ativacao_entrada in itertools.product(range(1, 4), ativacoes, ativacoes):
    # Gera todas as combinações possíveis de números de neurônios para as camadas ocultas
    combinacoes_neuronios = itertools.product(neuronios_oculta, repeat=num_camadas)
    
    # Itera sobre as combinações de neurônios por camada
    for neuronios_por_camada in combinacoes_neuronios:
        # Cria a rede neural
        modelo = criar_rede_neural(num_camadas, neuronios_por_camada, input_dim, output_dim, dropout_rate)
        
        # Define a função de ativação da camada de entrada
        modelo.layers[0].activation = tf.keras.activations.get(ativacao_entrada)
        
        # Define a função de ativação das camadas ocultas
        for i in range(1, num_camadas + 1):
            modelo.layers[i].activation = tf.keras.activations.get(ativacao_oculta)
        
        # Compila o modelo
        modelo.compile(optimizer='adam', loss='mean_squared_error')
        
        # Treina o modelo
        historico = modelo.fit(entradas_treinamento, saidas_treinamento, epochs=100, batch_size=32, validation_data=(entradas_validacao, saidas_validacao), verbose=0)
        
        # Avalia o modelo
        perda = modelo.evaluate(entradas_teste, saidas_teste)
        
        # Verifica se a perda atual é menor que a melhor perda encontrada até agora
        if perda < melhor_perda:
            melhor_perda = perda
            melhor_num_camadas = num_camadas
            melhor_ativacao_oculta = ativacao_oculta
            melhor_ativacao_entrada = ativacao_entrada
            melhor_neuronios_por_camada = neuronios_por_camada
            melhor_modelo = modelo
            melhor_historico = historico

# Imprime os resultados da melhor combinação encontrada
print('Melhor combinação:')
print('Número de camadas ocultas:', melhor_num_camadas)
print('Função de ativação das camadas ocultas:', melhor_ativacao_oculta)
print('Função de ativação da camada de entrada:', melhor_ativacao_entrada)
print('Número de neurônios por camada:', melhor_neuronios_por_camada)
print('Perda (Loss) do modelo:', melhor_perda)

# Gera gráfico da perda durante o treinamento da melhor combinação
plt.plot(melhor_historico.history['loss'])
plt.plot(melhor_historico.history['val_loss'])
plt.title('Perda do Modelo (Melhor Combinação)')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend(['Treinamento', 'Validação'])
plt.show()

# Gera a figura da rede neural da melhor combinação
tf.keras.utils.plot_model(melhor_modelo, to_file='rede_neural.png', show_shapes=True, show_layer_names=True)
'''
