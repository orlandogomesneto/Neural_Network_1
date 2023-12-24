import matplotlib.pyplot as plt

# Defina o número de entradas, neurônios intermediários e saídas
num_inputs = 4
num_hidden_neurons = 4
num_hidden_neurons2 = 0
num_outputs = 3

# Defina as posições das camadas
input_layer_x = 2.5
hidden_layer_x = 4
hidden_layer_x2 = 5.5
output_layer_x = 7

# Defina as alturas das camadas
layer_height = 0.8

# Nomes das entradas e saídas
input_names = ['pH', 'S.S.', 'D.O.', 'T']
output_names = ['X', 'S', 'P']

# Crie uma figura com alta qualidade
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

# Desenhe as camadas de entrada
for i, name in enumerate(input_names):
    ax.add_patch(plt.Circle((input_layer_x, i * layer_height), 0.2, color='black', fill=False))
    ax.text(input_layer_x - 0.6, i * layer_height, name, fontsize=18, verticalalignment='center')
    if i == 2:
        ax.text(input_layer_x - 0.18, i * layer_height - 2.0, 'inputs', fontsize=14, verticalalignment='center', color='gray')


# Desenhe as camadas intermediárias e adicione pontos entre o segundo e o terceiro neurônio
for i in range(num_hidden_neurons):
    ax.add_patch(plt.Circle((hidden_layer_x, i * layer_height), 0.2, color='black', fill=False))
    if i == 2:
        ax.text(hidden_layer_x - 0.04, i * layer_height * 0.86, '.', fontsize=28, verticalalignment='center_baseline')
        ax.text(hidden_layer_x - 0.04, i * layer_height * 0.80, '.', fontsize=28, verticalalignment='center_baseline')
        ax.text(hidden_layer_x - 0.04, i * layer_height * 0.74, '.', fontsize=28, verticalalignment='center_baseline')
        ax.text(hidden_layer_x - 0.04, i * layer_height * 1.5, '1', fontsize=14, verticalalignment='center')
        ax.text(hidden_layer_x - 0.04, i * layer_height * 1, '2', fontsize=14, verticalalignment='center')
        ax.text(hidden_layer_x - 0.04, i * layer_height * 0.5, 'n', fontsize=14, verticalalignment='center')
        ax.text(hidden_layer_x - 0.14, i * layer_height * 0.0, 'n+1', fontsize=14, verticalalignment='center')
        ax.text(hidden_layer_x + 0.28, i * layer_height - 2.0, 'camadas ocultas', fontsize=14, verticalalignment='center', color='gray')

# Desenhe as camadas intermediárias e adicione pontos entre o segundo e o terceiro neurônio
for i in range(num_hidden_neurons2):
    ax.add_patch(plt.Circle((hidden_layer_x2, i * layer_height), 0.2, color='black', fill=False))
    if i == 2:
        ax.text(hidden_layer_x2 - 0.04, i * layer_height * 0.86, '.', fontsize=28, verticalalignment='center_baseline')
        ax.text(hidden_layer_x2 - 0.04, i * layer_height * 0.80, '.', fontsize=28, verticalalignment='center_baseline')
        ax.text(hidden_layer_x2 - 0.04, i * layer_height * 0.74, '.', fontsize=28, verticalalignment='center_baseline')
        ax.text(hidden_layer_x2 - 0.04, i * layer_height * 1.5, '1', fontsize=14, verticalalignment='center')
        ax.text(hidden_layer_x2 - 0.04, i * layer_height * 1, '2', fontsize=14, verticalalignment='center')
        ax.text(hidden_layer_x2 - 0.04, i * layer_height * 0.5, 'n', fontsize=14, verticalalignment='center')
        ax.text(hidden_layer_x2 - 0.14, i * layer_height * 0.0, 'n+1', fontsize=14, verticalalignment='center')


# Desenhe as camadas de saída
for i, name in enumerate(output_names):
    ax.add_patch(plt.Circle((output_layer_x, i * layer_height + 0.4), 0.2, color='black', fill=False))
    ax.text(output_layer_x + 0.3, i * layer_height + 0.4, name, fontsize=20, verticalalignment='center')
    if i == 2:
        ax.text(output_layer_x - 0.2, i * layer_height - 2.0, 'outputs', fontsize=14, verticalalignment='center', color='gray')

# Conecte os neurônios com setas
for i in range(num_inputs):
    for j in range(num_hidden_neurons):
        ax.annotate("", xy=(input_layer_x + 0.2, i * layer_height), xytext=(hidden_layer_x - 0.2, j * layer_height),
                    arrowprops=dict(arrowstyle="<-", lw=0.5, color='gray'))
        
for i in range(num_hidden_neurons): # Conecta as camadas intermediárias
    for j in range(num_hidden_neurons2):
        ax.annotate("", xy=(hidden_layer_x2 - 0.2, i * layer_height), xytext=(hidden_layer_x + 0.2, j * layer_height),
                    arrowprops=dict(arrowstyle="->", lw=0.5, color='gray'))

for i in range(num_hidden_neurons2):
    for j in range(num_outputs):
        ax.annotate("", xy=(hidden_layer_x2 + 0.2, i * layer_height), xytext=(output_layer_x - 0.2, j * layer_height + 0.4),
                    arrowprops=dict(arrowstyle="<-", lw=0.5, color='gray'))

# Configura os limites dos eixos e exibe a figura
ax.set_xlim(1.8, 7.5)
ax.set_ylim(-0.5, num_hidden_neurons * layer_height - 0.4)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.savefig('IMAGEM TCC 2 2 .png', dpi=300)
plt.show()