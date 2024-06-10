import gmsh
from itertools import combinations
import numpy as np

# flake8: noqa: E501


class GeometryCleaner:
    def __init__(self):
        self.lines = []
        self.line_connectivity = {}
        self.lines_correspondence = {}
        self.lines_points = {}
        self.foster_lines = {}
        self.unique_correspondences = []
        self.unique_connectivities = []
        self.lonely_lines = []

    def analyze_common_points(self):
        self.get_lines_from_geometry()  # * OK
        self.get_line_connectivity()  # * OK
        self.get_foster_lines()
        # self.check_unique_connectivities()
        print("***")

    def get_lines_from_geometry(self):
        self.lines = []
        lines = gmsh.model.occ.getEntities(1)
        for line in lines:
            self.lines.append(line[1])

    def get_line_connectivity(self):
        for line in self.lines:
            _, points = gmsh.model.getAdjacencies(1, line)
            p1, p2 = points[0], points[1]
            self.line_connectivity[line] = [p1, p2]
        print(self.line_connectivity)
        print("***")

    def get_foster_lines(self):
        self.foster_lines = {}
        lines_combinations = combinations(self.line_connectivity.items(), 2)

        for (line1_id, line1_points), (line2_id, line2_points) in lines_combinations:
            if np.array_equal(np.sort(line1_points), np.sort(line2_points)):
                # Store the line IDs as a tuple in the dictionary
                self.foster_lines[(line1_id, line2_id)] = line1_points

        self.foster_lines_list = list(
            set([item for sublist in self.foster_lines.keys() for item in sublist])
        )

    def check_unique_connectivities(self):
        line_list = []
        for volume in gmsh.model.getEntities(3):
            _, surfaces = gmsh.model.getAdjacencies(3, volume[1])
            for surfs in surfaces:
                _, lines = gmsh.model.getAdjacencies(2, surfs)
                line_list.append(lines)
                for ll in line_list:
                    if ll not in self.foster_lines_list:
                        self.lonely_lines.append(ll)
        print("Done")
