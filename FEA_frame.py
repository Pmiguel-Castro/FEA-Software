import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve
import tkinter as tk
from fea.model.element import FrameElement
from fea.model.node import Node
from fea.model.material import Material

class FrameModel:
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.fixed_dofs = set()
        self.n_nodes = 0
        self.f_global = []
        pass

    def add_node(self, node):
        self.nodes.append(node)
        self.f_global.extend(np.zeros(3))
        pass

    def add_element(self, element):
        self.elements.append(element)
        pass

    def build_matrix(self):
        self.n_nodes = len(self.nodes)
        self.k_global = dok_matrix((3 * self.n_nodes, 3 * self.n_nodes))
        for element in self.elements:
            index = np.ix_(element.dof, element.dof)
            self.k_global[index] += element.k_local
        self.k_global = self.k_global.tocsr()
        pass

    def apply_force(self, node_id, f_x, f_y, m_z):
        self.f_global[3 * node_id] += f_x
        self.f_global[3 * node_id + 1] += f_y
        self.f_global[3 * node_id + 2] += m_z
        pass

    def apply_constraint(self, node_id, isfixed_x, isfixed_y, isfixed_theta):
        if isfixed_x:
            self.fixed_dofs.add(3 * node_id)
        if isfixed_y:
            self.fixed_dofs.add(3 * node_id + 1)
        if isfixed_theta:
            self.fixed_dofs.add(3 * node_id + 2)
        pass

    def solve(self):
        self.u_global = np.zeros(3 * self.n_nodes)
        free_dofs = [x for x in range(3 * self.n_nodes) if x not in self.fixed_dofs]
        k_free = self.k_global[np.ix_(free_dofs, free_dofs)]
        f_free = np.array(self.f_global)[free_dofs]
        u_free = spsolve(k_free, f_free)
        self.u_global[free_dofs] = u_free

        for node in self.nodes:
            node.ux = self.u_global[3 * node.id]
            node.uy = self.u_global[3 * node.id + 1]
            node.utheta = self.u_global[3 * node.id + 2]
        pass

    def plot_frame(self):
        for element in self.elements:
            normal_force = element.get_int_forces(self.u_global)[3]
            if normal_force > 1.0: 
                bar_color = 'red'
            elif normal_force < -1.0: 
                bar_color = 'blue'
            else: 
                bar_color = 'gray'
            plt.plot([element.start.x, element.end.x], [element.start.y, element.end.y], linestyle = '--', marker='o', color = 'black')
            plt.plot([element.start.x_deformed, element.end.x_deformed], [element.start.y_deformed, element.end.y_deformed], linestyle = '--', marker='o', color = bar_color)
        plt.axis('equal')
        plt.show()

class FEA_app:
    def __init__(self, root, model):
        self.model = model
        self.entry_x = tk.Entry(root)
        self.entry_y = tk.Entry(root)
        self.entry_x.pack()
        self.entry_y.pack()
        tk.Button(root, text="Adcionar Nó", command=self.gui_add_node).pack()
        tk.Button(root, text="Ver Estrutura", command=self.gui_plot).pack()
        self.count_nodes = 0
        pass

    def gui_add_node(self):
        x = float(self.entry_x.get())
        y = float(self.entry_y.get())
        self.model.add_node(Node(self.count_nodes, x, y))
        self.count_nodes += 1
        pass

    def gui_plot(self):
        if len(self.model.nodes) == 0:
            return
        x_coords = []
        y_coords = []
        for node in self.model.nodes:
            x_coords.append(node.x)
            y_coords.append(node.y)
        plt.scatter(x_coords, y_coords, color='black', marker='o')
        plt.show()

#window = tk.Tk()
#window.title("FEA Software")
#window.geometry("720x480")

#motor = FrameModel()
#app = FEA_app(window, motor)
#window.mainloop()

node1 = Node(0, 0, 0)
node2 = Node(1, 1, 0)
node3 = Node(2, 0.5, 0.5)

steel = Material(7.1*(10**10), 0.33)

modelo_teste = FrameModel()
modelo_teste.add_node(node1)
modelo_teste.add_node(node2)
modelo_teste.add_node(node3)
modelo_teste.add_element(FrameElement(node1, node2, 0.1, 0.1, steel))
modelo_teste.add_element(FrameElement(node1, node3, 0.1, 0.1, steel))
modelo_teste.add_element(FrameElement(node2, node3, 0.1, 0.1, steel))

modelo_teste.build_matrix()

modelo_teste.apply_constraint(0, True, True, False)
modelo_teste.apply_constraint(1, False, True, False)
modelo_teste.apply_force(2, 0, -5*(10**7), 0)

modelo_teste.solve()

for element in modelo_teste.elements:
    print(f"{element.get_int_forces(modelo_teste.u_global)}\n")

modelo_teste.plot_frame()
print(f"ux: {modelo_teste.u_global[6]}\nuy: {modelo_teste.u_global[7]}\ntheta: {modelo_teste.u_global[8]}")

