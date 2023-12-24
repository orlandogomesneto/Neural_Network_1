import numpy as np
import matplotlib.pyplot as plt

def plot_functions():
    # Definindo o domínio dos valores de x
    x = np.linspace(-5, 5, 400)

    # Calculando os valores das funções
    logsig = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    linear = x

    # Função para configurar a aparência de cada gráfico
    def configure_plot(ax, title):
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_title(title)

    # Gráfico da função logsig
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, logsig, label="Logsig", color="blue", linewidth=2)
    configure_plot(ax, "Função Logsig")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("logsig.png", dpi=300)
    plt.close()

    # Gráfico da função tangente hiperbólica
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, tanh, label="Tangente Hiperbólica", color="green", linewidth=2)
    configure_plot(ax, "Função Tansig")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("tanh.png", dpi=300)
    plt.close()

    # Gráfico da função linear
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, linear, label="Linear", color="red", linewidth=2)
    configure_plot(ax, "Função purelin")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("linear.png", dpi=300)
    plt.close()

# Chamando a função para gerar e salvar os gráficos
plot_functions()
