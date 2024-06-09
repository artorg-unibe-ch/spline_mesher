import gmsh
from itertools import combinations


# flake8: noqa: E501


class GeometryCleaner:
    def __init__(self):
        self.line_connectivity = {}
        self.lines_to_remove = []
        self.surfaces_to_remove = set()
        self.volumes_to_remove = set()
        self.bsplines_thrusection = []
        self.lines_thrusection = []

    def analyze_geometry(self):
        self.get_lines_from_geometry()
        self.get_lines_from_thru_section()
        self.get_line_connectivity()
        self.identify_lines_to_remove()
        self.identify_surfaces_and_volumes_to_remove()

    def get_lines_from_geometry(self):
        self.lines_rectangles = []
        for rect in gmsh.model.occ.getEntities(2):
            _, lines = gmsh.model.getAdjacencies(dim=2, tag=rect[1])
            self.lines_rectangles.extend(lines)

    def get_lines_from_thru_section(self):
        self.lines_thrusection = []
        self.bsplines_thrusection = []
        for volume in gmsh.model.getEntities(3):
            _, surfs = gmsh.model.getAdjacencies(dim=3, tag=volume[1])
            for surf in surfs:
                _, lines = gmsh.model.getAdjacencies(dim=2, tag=surf)
                for line in lines:
                    if gmsh.model.getCurvature(1, line, [0.5])[0] > 1e-14:
                        self.bsplines_thrusection.append(line)
                    else:
                        self.lines_thrusection.append(line)
        self.lines_thrusection = sorted(list(set(self.lines_thrusection)))

    def get_line_connectivity(self):
        for line in self.lines_thrusection:
            points = gmsh.model.getBoundary([(1, line)])
            p1, p2 = points[0][1], points[1][1]
            x0, y0, z0 = gmsh.model.getValue(0, p1, [])
            x1, y1, z1 = gmsh.model.getValue(0, p2, [])
            self.line_connectivity[line] = [(x0, y0, z0), (x1, y1, z1), p1, p2]

    def check_rectangle(self, lines):
        points = []
        for line in lines:
            p1, p2 = self.line_connectivity[line][2], self.line_connectivity[line][3]
            points.append(p1)
            points.append(p2)
        unique_points = set(points)
        if len(unique_points) != 4:
            return False
        z_coords = [self.line_connectivity[line][0][2] for line in lines] + [
            self.line_connectivity[line][1][2] for line in lines
        ]
        return len(set(z_coords)) == 1

    def identify_lines_to_remove(self):
        line_combinations = list(combinations(self.lines_thrusection, 4))
        for line_set in line_combinations:
            if self.check_rectangle(line_set):
                self.lines_to_remove.extend(line_set)

        # in self.bsplines_thrusection, keep only those that are also part of self.lines_thrusection
        self.bsplines_thrusection = [
            line
            for line in self.bsplines_thrusection
            if line not in self.lines_thrusection
        ]
        self.lines_to_remove = list(set(self.lines_to_remove))

    def identify_surfaces_and_volumes_to_remove(self):
        for line in self.lines_to_remove:
            surfaces, _ = gmsh.model.getAdjacencies(1, line)
            for surface in surfaces:
                self.surfaces_to_remove.add(surface)
                volumes, _ = gmsh.model.getAdjacencies(2, surface)
                for volume in volumes:
                    self.volumes_to_remove.add(volume)
