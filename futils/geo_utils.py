import re
# TODO: enhancements and correct implementation required to class GeoSort
# label: enhancement
# assignee: @simoneponcioni
# milestone: 0.1.0

class GeoSort():
    def __init__(self, file1, file2, filename):
        self.file1 = str(file1)
        self.file2 = str(file2)
        self.filename = str(filename)
        self.filename_sorted = str(filename + '_sorted.geo_unrolled')

    def append_file2_to_file1(self, file1, file2):
        with open(file1, 'a') as outfile:
            for line in self.sort_lines(file2):
                outfile.write(line)
        return None

    def sort_lines(filename):
        with open(filename) as f:
            return f.readlines()

    def read_lines(self):
        file = self.sort_lines(self.filename)
        # TODO: cl_s already includes points and only 'cl__1'
        # label: bug
        # assignees: @simoneponcioni
        # milestone: 0.1.0
        cl_s = [line for line in file if 'cl__1' in line]
        points = [line for line in file if 'Point' in line]
        lines = [line for line in file if 'Line' in line]
        CurveLoops = [line for line in file if re.search(r'\bCurve Loop\b', line)]
        Surfaces = [line for line in file if 'Surface' in line]
        PlaneSurfaces = [line for line in file if re.search(r'\bPlane Surface\b', line)]
        SurfaceLoops = [line for line in file if re.search(r'\bSurface Loop\b', line)]
        Volumes = [line for line in file if 'Volume' in line]
        bool_gmsh = [f'BooleanDifference{{ Volume{{{Volumes[0].split("(")[1].split(")")[0]}}}; Delete;}}{{ Volume{{{Volumes[1].split("(")[1].split(")")[0]}}}; Delete;}}']
        return ['SetFactory("OpenCASCADE");\n', cl_s, lines, CurveLoops, Surfaces, PlaneSurfaces, SurfaceLoops, Volumes, bool_gmsh]

    def write_geo(self, filename_sorted):
        with open(filename_sorted, 'w') as outfile:
            for entity in self.read_lines():
                for elem in entity:
                    outfile.write(elem)
        return None