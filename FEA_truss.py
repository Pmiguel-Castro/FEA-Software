import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve
from fea.model.element import TrussElement as Element

class TrussModel:
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.fixed_dofs = []
        self.n_nodes = 0
        pass

    def add_node(self, node):
        self.nodes.append(node)
        pass

    def add_element(self, element):
        self.elements.append(element)
        pass

    def build_matrix(self):
        self.n_nodes = len(self.nodes)
        self.k_global = dok_matrix((2 * self.n_nodes, 2 * self.n_nodes))
        for element in self.elements:
            index = np.ix_(element.dof, element.dof)
            self.k_global[index] += element.k_local
        self.k_global = self.k_global.tocsr()
        self.f_global = np.zeros(2 * self.n_nodes)
        pass

    def apply_force(self, node_id, f_x, f_y):
        self.f_global[2 * node_id] += f_x
        self.f_global[2 * node_id + 1] += f_y
        pass

    def apply_constraint(self, node_id, isfixed_x, isfixed_y):
        if isfixed_x:
            self.fixed_dofs.append(2 * node_id)
        if isfixed_y:
            self.fixed_dofs.append(2 * node_id + 1)
        pass

    def solve(self):
        self.u_global = np.zeros(2 * self.n_nodes)
        free_dofs = [x for x in range(2 * self.n_nodes) if x not in self.fixed_dofs]
        k_free = self.k_global[np.ix_(free_dofs, free_dofs)]
        f_free = self.f_global[free_dofs]
        u_free = spsolve(k_free, f_free)
        self.u_global[free_dofs] = u_free
        for node in self.nodes:
            node.x_def += self.u_global[2 * node.id]
            node.y_def += self.u_global[2 * node.id + 1]
        pass

    def plot_truss(self):
        for element in self.elements:
            plt.plot([element.start.x, element.end.x], [element.start.y, element.end.y], linestyle = '--', color = 'blue')
            plt.plot([element.start.x_def, element.end.x_def], [element.start.y_def, element.end.y_def], linestyle = '--', marker='o', color = 'red')
        plt.axis('equal')
        plt.show()

node1 = Node(0, 0, 0)
node2 = Node(1, 1, 0)
node3 = Node(2, 0.5, 0.5)

steel = Material(7.1*(10**10), 0.33)

modelo_teste = TrussModel()
modelo_teste.add_node(node1)
modelo_teste.add_node(node2)
modelo_teste.add_node(node3)
modelo_teste.add_element(Element(node1, node2, 0.1, 0.1, steel))
modelo_teste.add_element(Element(node1, node3, 0.1, 0.1, steel))
modelo_teste.add_element(Element(node2, node3, 0.1, 0.1, steel))

modelo_teste.build_matrix()

modelo_teste.apply_constraint(0, True, True)
modelo_teste.apply_constraint(1, False, True)
modelo_teste.apply_force(2, 0, -5*(10**7))

modelo_teste.solve()
modelo_teste.plot_truss()
print(f"ux: {modelo_teste.u_global[4]}\nuy: {modelo_teste.u_global[5]}")
