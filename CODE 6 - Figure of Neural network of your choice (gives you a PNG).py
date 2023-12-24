import matplotlib.pyplot as plt

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, input_labels, output_labels):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Input arrows and labels
    layer_top = v_spacing*(layer_sizes[0] - 1)/2. + (top + bottom)/2.
    for m, label in enumerate(input_labels):
        line = plt.Line2D([left - 0.2*h_spacing, left], 
                          [layer_top - m*v_spacing, layer_top - m*v_spacing], c='m')
        ax.add_artist(line)
        ax.text(left - 0.23*h_spacing, layer_top - m*v_spacing, label, ha='right', va='center', fontsize=25)  # Adjust fontsize here
    
    # Output arrows and labels
    layer_top = v_spacing*(layer_sizes[-1] - 1)/2. + (top + bottom)/2.
    for m, label in enumerate(output_labels):
        line = plt.Line2D([right, right + 0.2*h_spacing], 
                          [layer_top - m*v_spacing, layer_top - m*v_spacing], c='m')
        ax.add_artist(line)
        ax.text(right + 0.23*h_spacing, layer_top - m*v_spacing, label, ha='left', va='center', fontsize=25)  # Adjust fontsize here
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/8., # O DIVISOR MUDA O TAMANHO DO PONTO
                                color='w', ec='m', zorder=4)
            ax.add_artist(circle)
    
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='g')
                ax.add_artist(line)

fig = plt.figure(figsize=(12, 12))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .01, .99, [2, 3, 1], ['P', 'T'], ['V'])
fig.tight_layout()
fig.savefig('REDE TIAGO 2-3-1.png')
