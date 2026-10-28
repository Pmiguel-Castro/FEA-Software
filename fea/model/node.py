class Node:
    def __init__(self, id_no, coord_x, coord_y):
        self.id = id_no
        self.x = coord_x
        self.y = coord_y
        self.dof = [self.id * 3, self.id * 3 + 1, self.id * 3 + 2]
        self.ux = 0
        self.uy = 0 
        self.utheta = 0
        pass

    @property    
    def x_deformed(self):
        return self.x + self.ux
    @property
    def y_deformed(self):
        return self.y + self.uy