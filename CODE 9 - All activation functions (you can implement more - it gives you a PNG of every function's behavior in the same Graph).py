import numpy as np
import matplotlib.pyplot as plt

def plot_functions():
    # Definindo o domínio dos valores de x
    x = np.linspace(-2, 2, 1200)

    # Calculando os valores das funções
    logsig = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    linear = x
    softmax = np.exp(x) / np.sum(np.exp(x), axis=0)
    relu = np.maximum(0, x)

    # Função para configurar a aparência de cada gráfico
    def configure_plot(ax, title):
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_title(title)

    # Criando um único gráfico com todas as funções
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, logsig, label="Logsig", color="blue", linewidth=2)
    ax.plot(x, tanh, label="Tansig", color="green", linewidth=2)
    ax.plot(x, linear, label="purelin", color="red", linewidth=2)

    configure_plot(ax, " ")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

# Chamando a função para gerar o gráfico combinado
plot_functions()
plt.savefig('func.png', dpi=600)
plt.show()