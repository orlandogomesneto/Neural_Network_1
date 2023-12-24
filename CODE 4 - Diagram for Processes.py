import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(30, 5), dpi=600)
ax.axis('off')
block_width = 0.1
block_spacing = 0.15
arrow_length = 0.2
block_y = 1
block_height = 0.015

processes = ['Preparo do Mosto', 'Fermentação', 'Descuba', 'Maturação', 'Transfega', 'Filtração', 'Pasteurização', 'Envase']
font_size = 10.5

for i, process in enumerate(processes):
    # Blocos
    rect = patches.Rectangle((i * block_spacing, block_y - block_height / 2), block_width, block_height, facecolor='skyblue', edgecolor='black')
    ax.add_patch(rect)
    ax.text(i * block_spacing + block_width / 2, 0.949*block_y - block_height / 2 + 0.06, process, ha='center', va='top', fontsize=font_size)
    
    # Setas
    if i < len(processes) - 1:
        arrow_x = (i * block_spacing) + block_width
        ax.arrow(arrow_x, block_y, arrow_length - 0.01, 0, head_width=0.005, head_length=0.009, fc='k', ec='k')

# Setas de entrada
inputs = ['Mel', 'Casca de Jabuticaba', 'Aditivos', 'Água']
for j, input_name in enumerate(inputs):
    ax.arrow(-0.092, block_y - 0.01 + j*0.006, 0.08, 0, head_width=0.003, head_length=0.009, fc='k', ec='k')
    ax.text(-0.098, block_y - 0.01 + j*0.006, input_name, ha='right', va='center', fontsize=14)

# Seta de entrada para leveduras
ax.arrow(block_spacing*1.3, block_y*1.072, 0.0, -0.055, head_width=0.01, head_length=0.009, fc='g', ec='g')
ax.text(block_spacing*1.6, block_y*0.83 + 0.2, 'Leveduras', ha='center', va='center', fontsize=14)

ax.text(len(processes) * block_spacing - block_spacing + 0.185, block_y + 0.0007, 'Hidromel', ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig('diagrama_fermentacao.png', dpi=300, bbox_inches='tight')  # Salvar com alta qualidade
plt.show()
