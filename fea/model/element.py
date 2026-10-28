import numpy as np

class Element:
    def __init__(self, init_node, end_node, height, width, material):
        self.start = init_node
        self.end = end_node
        self.area = height * width
        self.e = material.e
        self.length = ((self.start.x - self.end.x) ** 2 + 
                       (self.start.y - self.end.y) ** 2) ** (0.5)
        self.sin = (self.end.y - self.start.y) / self.length 
        self.cos = (self.end.x - self.start.x) / self.length
        self.dof = self.start.dof + self.end.dof
        pass

class TrussElement(Element):
    def __init__(self, init_node, end_node, height, width, material):
        super().__init__(init_node, end_node, height, width, material)
        self.k = self.e * self.area / self.length
        self.k_local = self.k * np.array([[self.cos ** 2, self.cos * self.sin, -self.cos ** 2, -self.cos * self.sin],
                                          [self.cos * self.sin, self.sin ** 2, -self.cos * self.sin, -self.sin ** 2],
                                          [-self.cos ** 2, -self.cos * self.sin, self.cos ** 2, self.cos * self.sin],
                                          [-self.cos * self.sin, -self.sin ** 2, self.cos * self.sin, self.sin ** 2]])
        pass

class FrameElement(Element):
    def __init__(self, init_node, end_node, height, width, material):
        super().__init__(init_node, end_node, height, width, material)
        self.v = material.v
        self.inertia = (width * height ** 3) / 12

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
        pass

    def get_int_forces(self, u_global):
        u_bar = u_global[self.dof]
        f_bar = self.k_bar @ self.t @ u_bar
        return f_bar