# Test Mesh Element Order with Gmsh
# Simone Poncioni, 01.05.2024

import gmsh


class TestMeshOrder:
    def __init__(self):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.option = gmsh.option
        self.plugin = gmsh.plugin
        self.element_order = int(1)

    def create_cube(self):
        gmsh.initialize()
        self.factory.addBox(0, 0, 0, 1, 1, 1)
        self.factory.synchronize()

        # * ######## HERE
        self.model.mesh.setOrder(self.element_order)
        # self.option.setNumber("Mesh.ElementOrder", self.element_order)
        # * ########

        self.model.mesh.generate(3)

        # print the actual element order
        meshed_order = self.option.getNumber("Mesh.ElementOrder")
        gmsh.finalize()
        assert self.element_order == meshed_order


def main():
    test_elm_order = TestMeshOrder()
    test_elm_order.create_cube()


if __name__ == "__main__":
    main()
