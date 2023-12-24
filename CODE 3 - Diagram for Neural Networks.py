from graphviz import Digraph

dot = Digraph('NeuralNetworkTraining', node_attr={'style': 'filled', 'shape': 'ellipse'})
dot.attr(size='10,10')

dot.attr('node', fontname='Arial', fontsize='12', fontcolor='black', penwidth='2')
dot.attr('edge', fontname='Arial', fontsize='12', fontcolor='black', color='gray50', arrowsize='0.7')

def add_node(name, label):
    dot.node(name, label, color='lightgreen:whitesmoke', gradientangle='270', penwidth='2.5')

add_node('A', 'Seleção de Variáveis de Entrada e Saída')
add_node('B', 'Experimentos para coleta de Dados')
add_node('C', 'Preparação e Tratamento dos Dados')
add_node('D', 'Configuração da Rede Neural')
add_node('E', 'Treinamento, Validação e Teste da Rede')
add_node('F', 'Simulação da Rede')
add_node('G', 'Equacionamento e Análise dos Resultados')

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])

dot.format = 'png'
dot.renderer = 'cairo'
dot.formatter = 'cairo'
dot.attr(dpi='300')
dot.render('neural_network_training_diagram.png', view=True)
