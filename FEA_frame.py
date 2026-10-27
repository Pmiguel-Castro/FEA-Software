import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve


class Material:
    def __init__(self, young_modulus, poisson_ratio):
        self.e = young_modulus
        self.v = poisson_ratio
        pass

class Node:
    def __init__(self, id_no, coord_x, coord_y):
        self.id = id_no
        self.x = coord_x
        self.y = coord_y
        self.dof = [self.id * 3, self.id * 3 + 1, self.id * 3 + 2]
        self.x_def = self.x
        self.y_def = self.y
        self.theta_def = 0.0
        pass

class FrameElement:
    def __init__(self, init_node, end_node, height, width, material):
        self.start = init_node
        self.end = end_node
        self.area = height * width
        self.e = material.e
        self.v = material.v
        self.inertia = (width * height ** 3) / 12
        self.length = ((self.start.x - self.end.x) ** 2 + 
                       (self.start.y - self.end.y) ** 2) ** (0.5)
        self.sin = (self.end.y - self.start.y) / self.length 
        self.cos = (self.end.x - self.start.x) / self.length
        self.k = self.e * self.area / self.length
        
        EA_L = self.e * self.area / self.length
        EI_L = 4 * self.e * self.inertia / self.length
        EI_L_2 = 2 * self.e * self.inertia / self.length
        EI_L2 = 6 * self.e * self.inertia / (self.length ** 2)
        EI_L3 = 12 * self.e * self.inertia / (self.length ** 3)

        self.k_bar = np.array([[ EA_L,   0  ,    0  , -EA_L,   0  ,    0  ],
                          [  0  , EI_L3,  EI_L2,   0  ,-EI_L3,  EI_L2],
                          [  0  , EI_L2,  EI_L ,   0  ,-EI_L2, EI_L_2],
                          [-EA_L,   0  ,    0  ,  EA_L,   0  ,    0  ],
                          [  0  ,-EI_L3, -EI_L2,   0  , EI_L3, -EI_L2],
                          [  0  , EI_L2, EI_L_2,   0  ,-EI_L2,  EI_L ]])
        self.t = np.array([[ self.cos, self.sin, 0 ,    0    ,    0    , 0],
                      [-self.sin, self.cos, 0 ,    0    ,    0    , 0],
                      [    0    ,    0    , 1 ,    0    ,    0    , 0],
                      [    0    ,    0    , 0 , self.cos, self.sin, 0],
                      [    0    ,    0    , 0 ,-self.sin, self.cos, 0],
                      [    0    ,    0    , 0 ,    0    ,    0    , 1]])
        
        self.k_local = self.t.T @ self.k_bar @ self.t
        self.dof = self.start.dof + self.end.dof
        pass

    def get_int_forces(self, u_global):
        u_bar = u_global[self.dof]
        f_bar = self.k_bar @ self.t @ u_bar
        return f_bar

class FrameModel:
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
        self.k_global = dok_matrix((3 * self.n_nodes, 3 * self.n_nodes))
        for element in self.elements:
            index = np.ix_(element.dof, element.dof)
            self.k_global[index] += element.k_local
        self.k_global = self.k_global.tocsr()
        self.f_global = np.zeros(3 * self.n_nodes)
        pass

    def apply_force(self, node_id, f_x, f_y, m_z):
        self.f_global[3 * node_id] += f_x
        self.f_global[3 * node_id + 1] += f_y
        self.f_global[3 * node_id + 2] += m_z
        pass

    def apply_constraint(self, node_id, isfixed_x, isfixed_y, isfixed_theta):
        if isfixed_x:
            self.fixed_dofs.append(3 * node_id)
        if isfixed_y:
            self.fixed_dofs.append(3 * node_id + 1)
        if isfixed_theta:
            self.fixed_dofs.append(3 * node_id + 2)
        pass

    def solve(self):
        self.u_global = np.zeros(3 * self.n_nodes)
        free_dofs = [x for x in range(3 * self.n_nodes) if x not in self.fixed_dofs]
        k_free = self.k_global[np.ix_(free_dofs, free_dofs)]
        f_free = self.f_global[free_dofs]
        u_free = spsolve(k_free, f_free)
        self.u_global[free_dofs] = u_free
        for node in self.nodes:
            node.x_def += self.u_global[3 * node.id]
            node.y_def += self.u_global[3 * node.id + 1]
            node.theta_def += self.u_global[3 * node.id + 2]
        pass

    def plot_frame(self):
        for element in self.elements:
            normal_force = element.get_int_forces(modelo_teste.u_global)[3]
            if normal_force > 1.0: 
                bar_color = 'red'
            elif normal_force < -1.0: 
                bar_color = 'blue'
            else: 
                bar_color = 'gray'
            plt.plot([element.start.x, element.end.x], [element.start.y, element.end.y], linestyle = '--', marker='o', color = 'black')
            plt.plot([element.start.x_def, element.end.x_def], [element.start.y_def, element.end.y_def], linestyle = '--', marker='o', color = bar_color)
        plt.axis('equal')
        plt.show()

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
#print(f"ux: {modelo_teste.u_global[6]}\nuy: {modelo_teste.u_global[7]}\ntheta: {modelo_teste.u_global[8]}")

