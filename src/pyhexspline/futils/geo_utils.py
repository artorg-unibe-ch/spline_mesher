import re
import shutil


class GeoSort:
    def __init__(self, file1, file2, filename, boolean):
        self.file1 = str(file1)
        self.file2 = str(file2)
        self.filename = str(filename)
        self.filename_sorted = str(filename + "_sorted.geo_unrolled")
        self.boolean = str(boolean)

    def append_file2_to_file1(self, file1, file2):
        with open(self.filename_sorted, "wb") as wfd:
            for f in [file1, file2]:
                with open(f, "rb") as fd:
                    shutil.copyfileobj(fd, wfd)

    def sort_lines(self, filename):
        with open(filename) as f:
            return f.readlines()

    def read_lines(self):
        file = self.sort_lines(self.filename_sorted)
        cl__s = [line for line in file if "cl__" in line.split("(")[0]]
        cl__s = list(set(cl__s))
        points = [line for line in file if "Point" in line]
        lines = [line for line in file if "Line" in line]
        CurveLoops = [line for line in file if re.search(r"\bCurve Loop\b", line)]
        Surfaces = [line for line in file if re.search(r"^Surface\(", line)]
        PlaneSurfaces = [line for line in file if re.search(r"\bPlane Surface\b", line)]
        SurfaceLoops = [line for line in file if re.search(r"\bSurface Loop\b", line)]
        Volumes = [line for line in file if "Volume" in line]
        bool_gmsh = [
            f'BooleanDifference{{ Volume{{{Volumes[0].split("(")[1].split(")")[0]}}}; {self.boolean};}}{{ Volume{{{Volumes[1].split("(")[1].split(")")[0]}}}; {self.boolean};}}'
        ]
        return [
            'SetFactory("OpenCASCADE");\n',
            cl__s,
            points,
            lines,
            CurveLoops,
            Surfaces,
            PlaneSurfaces,
            SurfaceLoops,
            Volumes,
            bool_gmsh,
        ]

    def write_geo(self):
        rearranged_list = self.read_lines()
        with open(self.filename_sorted, "w") as outfile:
            for entity in rearranged_list:
                for elem in entity:
                    outfile.write(elem)
        return self.filename_sorted
