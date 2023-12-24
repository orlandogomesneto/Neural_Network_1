from graphviz import Digraph

dot = Digraph(format='pdf')


node_style = {'fontcolor': 'black'}
edge_style = {'fontcolor': 'black'}


nodes = [
    ('A1', 'Seleção da biomassa adequada', 'ellipse', 'peru'),
    ('A2', 'Cavitação Hidrodinâmica', 'box', 'lightcoral'),
    ('A3', 'Biomassa decomposta', 'ellipse', 'lightblue'),
    ('B', 'Açúcares simples', 'ellipse', 'lightblue'),
    ('C', 'Bioetanol Aquoso', 'ellipse', 'lightgreen'),
    ('D', 'Lignina e demais subprod.', 'ellipse', 'gray'),
    ('E', 'Bioetanol purificado', 'ellipse', 'lightgreen'),
    ('F', 'Vinhaça', 'ellipse', 'gray')
]

for n, label, shape, color in nodes:
    dot.node(n, label, shape=shape, color=color, style='filled', **node_style)


edges = [
    ('A1', 'A2', ' Pré-tratamento', 'peru'),
    ('A2', 'A3', ' Fragmentação celular', 'lightblue'),
    ('A3', 'B', ' Hidrólise enzimática', 'lightblue'),
    ('B', 'C', ' Fermentação alcoólica', 'lightgreen'),
    ('B', 'D', ' Resíduos (fermentação)', 'gray'),
    ('C', 'E', ' Destilação', 'lightgreen'),
    ('C', 'F', ' Resíduos (destilação)', 'gray')
]

for src, dst, label, color in edges:
    dot.edge(src, dst, label=label, color=color, **edge_style)


dot.render('Prod._Bioetanol_Cav.Hidrodinamica', view=True)
