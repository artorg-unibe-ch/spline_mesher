import gmsh
from pathlib import Path


class Mesher:
    def __init__(self, geo_file_path, mesh_file_path):
        self.model = gmsh.model
        self.factory = self.model.occ
        self.geo_file_path = geo_file_path
        self.mesh_file_path = mesh_file_path

    def build_msh(self):
        modelname = Path(self.geo_file_path).stem
        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add(modelname)
        gmsh.merge(self.geo_file_path)
        gmsh.geo.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.write(self.mesh_file_path)
        gmsh.finalize
