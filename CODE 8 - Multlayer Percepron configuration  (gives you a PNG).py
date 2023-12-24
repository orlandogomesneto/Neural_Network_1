import matplotlib.pyplot as plt

# Defina o número de entradas, neurônios intermediários e saídas
num_inputs = 3
num_hidden_neurons = 2
num_outputs = 1

# Defina as posições das camadas
input_layer_x = 2
hidden_layer_x = 3.5
output_layer_x = 4.5

# Defina as alturas das camadas
layer_height = 0.8

# para o Perceptron
input_names = ['', '', '']
output_names = ['']

# Crie uma figura com alta qualidade
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Desenhe as camadas de entrada
ax.add_patch(plt.Circle((input_layer_x, 1 * 0.0), 0.2, color='black', fill=False))
ax.text(input_layer_x - 0.4, 1 * 0.0, input_names[0], fontsize=18, verticalalignment='center')
ax.add_patch(plt.Circle((input_layer_x, 1 * 0.8), 0.2, color='black', fill=False))
ax.text(input_layer_x - 0.6, 1 * 0.8, input_names[1], fontsize=18, verticalalignment='center')
ax.add_patch(plt.Circle((input_layer_x, 1 * 1.6), 0.2, color='black', fill=False))
ax.text(input_layer_x - 0.6, 1 * 1.6, input_names[2], fontsize=18, verticalalignment='center')
ax.text(hidden_layer_x -1.7, 1 * 0.0 - 0.3, 'entradas', fontsize=14, verticalalignment='center', color='gray')

# Desenhe as camadas intermediárias e adicione pontos entre o segundo e o terceiro neurônio
ax.add_patch(plt.Circle((hidden_layer_x, 1 * 0.4), 0.2, color='black', fill=False))
ax.text(hidden_layer_x - 0.35, 1 * 0.0 - 0.3, 'camada oculta', fontsize=14, verticalalignment='center', color='gray')
ax.add_patch(plt.Circle((hidden_layer_x, 1 * 1.2), 0.2, color='black', fill=False))


# Desenhe as camadas de saída
ax.add_patch(plt.Circle((output_layer_x, 1 * layer_height), 0.2, color='black', fill=False))
ax.text(output_layer_x + 0.3, 1 * layer_height, output_names[0], fontsize=20, verticalalignment='center')
ax.text(output_layer_x - 0.1, 1 * 0.0 -0.3, 'saída', fontsize=14, verticalalignment='center', color='gray')

# Conecte os neurônios com setas
for i in range(num_inputs):
    for j in range(num_hidden_neurons):
        ax.annotate("", xy=(input_layer_x + 0.2, i * layer_height), xytext=(hidden_layer_x - 0.2, j * 0.8 + 0.4),
                    arrowprops=dict(arrowstyle="<-", lw=0.5, color='gray'))

for i in range(num_hidden_neurons):
    for j in range(num_outputs):
        ax.annotate("", xy=(hidden_layer_x + 0.2, i * 0.8 + 0.4), xytext=(output_layer_x - 0.2, j * 0.8 + 0.8),
                    arrowprops=dict(arrowstyle="<-", lw=0.5, color='gray'))

# Configura os limites dos eixos e exibe a figura
ax.set_xlim(1.7, 4.8)
ax.set_ylim(-0.5, num_hidden_neurons * layer_height + 0.3)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig('IMAGEM TCC PERCEPTRON.png', dpi=600)
plt.show()