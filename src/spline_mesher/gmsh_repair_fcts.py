# import trimesh
# import pymeshfix
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import struct
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import time

# TODO: write into class
# TODO: generalisation


def vtk2mhd(imvtk, spacing, filename, header=None):
    """
    writes a metaImage from a vtk array
    """
    writer = vtk.vtkMetaImageWriter()
    # different name depending on your vtk version
    try:
        writer.SetInputData(imvtk)
    except:
        writer.SetInput(imvtk)
    writer.SetFileName(filename)
    writer.Write()
    # corrects the wrong spacing written by vtkMetaImageWriter
    print("     "+filename)
    f = open(filename, 'r')
    content = f.read()
    f.close()
    f = open(filename, 'w')
    f.write(content.replace("ElementSpacing = 1 1 1", "ElementSpacing = "+str(spacing[2])+" "+str(spacing[1])+" "+str(spacing[0])))
    f.close()
    # writes AIM header if provided
    if header != None:
        f = open(filename, 'a')
        f.write("\n!-------------------------------------------------------------------------------\n")
        f.write("                                   AIM Log                                       ")
        f.write("\n!-------------------------------------------------------------------------------\n")
        for line in header:
            f.write(line+"\n")
        f.close()


def get_AIM_ints(f):
    """Function by Glen L. Niebur, University of Notre Dame (2010)
    reads the integer data of an AIM file to find its header length"""
    nheaderints = 32
    nheaderfloats = 8
    f.seek(0)
    binints = f.read(nheaderints * 4)
    header_int = struct.unpack("=32i", binints)
    return header_int


def AIMreader(fileINname, Spacing):
    """reads an AIM file and provides the corresponding vtk image with spacing, calibration data and header"""
    # read header
    print("     " + fileINname)
    with open(fileINname, 'rb') as f:
        AIM_ints = get_AIM_ints(f)
        # check AIM version
        if int(AIM_ints[5]) == 16:
            print("     -> version 020")
            if int(AIM_ints[10]) == 131074:
                format = "short"
                print("     -> format " + format)
            elif int(AIM_ints[10]) == 65537:
                format = "char"
                print("     -> format " + format)
            elif int(AIM_ints[10]) == 1376257:
                format = "bin compressed"
                print("     -> format " + format + " not supported! Exiting!")
                exit(1)
            else:
                format = "unknown"
                print("     -> format " + format + "! Exiting!")
                exit(1)
            header = f.read(AIM_ints[2])
            header_len = len(header) + 160
            extents = (0, AIM_ints[14] - 1, 0, AIM_ints[15] - 1, 0, AIM_ints[16] - 1)
        else:
            print("     -> version 030")
            if int(AIM_ints[17]) == 131074:
                format = "short"
                print("     -> format " + format)
            elif int(AIM_ints[17]) == 65537:
                format = "char"
                print("     -> format " + format)
            elif int(AIM_ints[17]) == 1376257:
                format = "bin compressed"
                print("     -> format " + format + " not supported! Exiting!")
                exit(1)
            else:
                format = "unknown"
                print("     -> format " + format + "! Exiting!")
                exit(1)
            header = f.read(AIM_ints[8])
            header_len = len(header) + 280
            extents = (0, AIM_ints[24] - 1, 0, AIM_ints[26] - 1, 0, AIM_ints[28] - 1)

    # collect data from header if existing
    # header = re.sub('(?i) +', ' ', header)
    header = header.split('\n'.encode())
    header.pop(0)
    header.pop(0)
    header.pop(0)
    header.pop(0)
    Scaling = None
    Slope = None
    Intercept = None
    IPLPostScanScaling = 1
    for line in header:
        if line.find(b"Orig-ISQ-Dim-p") > -1:
            origdimp = ([int(s) for s in line.split(b" ") if s.isdigit()])

        if line.find("Orig-ISQ-Dim-um".encode()) > -1:
            origdimum = ([int(s) for s in line.split(b" ") if s.isdigit()])

        if line.find("Orig-GOBJ-Dim-p".encode()) > -1:
            origdimp = ([int(s) for s in line.split(b" ") if s.isdigit()])

        if line.find("Orig-GOBJ-Dim-um".encode()) > -1:
            origdimum = ([int(s) for s in line.split(b" ") if s.isdigit()])

        if line.find("Scaled by factor".encode()) > -1:
            Scaling = float(line.split(" ".encode())[-1])
        if line.find("Density: intercept".encode()) > -1:
            Intercept = float(line.split(" ".encode())[-1])
        if line.find("Density: slope".encode()) > -1:
            Slope = float(line.split(" ".encode())[-1])
            # if el_size scale was applied, the above still takes the original voxel size. This function works
            # only if an isotropic scaling was applied!!!!
        if line.find("scale".encode()) > -1:
            IPLPostScanScaling = float(line.split(" ".encode())[-1])
    # Spacing is calculated from Original Dimensions. This is wrong, when the images were coarsened and
    # the voxel size is not anymore corresponding to the original scanning resolution!

    try:
        Spacing = IPLPostScanScaling * (
            np.around(np.asarray(origdimum) / np.asarray(origdimp) / 1000, 5)
        )
    except:
        pass
    # read AIM
    reader = vtk.vtkImageReader2()
    reader.SetFileName(fileINname)
    reader.SetDataByteOrderToLittleEndian()
    reader.SetFileDimensionality(3)
    reader.SetDataExtent(extents)
    reader.SetHeaderSize(header_len)
    if format == "short":
        reader.SetDataScalarTypeToShort()
    elif format == "char":
        reader.SetDataScalarTypeToChar()
    reader.SetDataSpacing(Spacing)
    reader.Update()
    imvtk = reader.GetOutput()
    return imvtk, Spacing, Scaling, Slope, Intercept, header


def read_aim(name, bone):
    """
    read AIM image
    --------------
    All necessary AIM files are imported and stored in bone dict
    Input: name specifier, filenames dict, bone dict
    Output: bone dict
    - numpy array containing AIM image
    """

    print(f" ... Reading file:\n{name}")

    Spacing = bone["Spacing"]
    # Read image as vtk
    # IMG_vtk = AIMreader(filenames[name + 'name'], Spacing)[0]
    IMG_vtk = AIMreader(name, Spacing)[0]
    # convert AIM files to numpy arrays
    IMG_array = vtk2numpy(IMG_vtk)
    np.save('imnp.npy', IMG_array)
    return bone, IMG_vtk


def vtk2numpy(imvtk):
    """turns a vtk image data into a numpy array"""
    dim = imvtk.GetDimensions()
    data = imvtk.GetPointData().GetScalars()
    imnp = vtk_to_numpy(data)
    # vtk and numpy have different array conventions
    imnp = imnp.reshape(dim[2], dim[1], dim[0])
    imnp = imnp.transpose(2, 1, 0)
    return imnp


def numpy2vtk(imnp, spacing):
    """turns a numpy array into a vtk image data"""
    # vtk and numpy have different array conventions
    imnp_flat = imnp.transpose(2, 1, 0).flatten()
    if imnp.dtype == "int8":
        arraytype = vtk.VTK_CHAR
    elif imnp.dtype == "int16":
        arraytype = vtk.VTK_SHORT
    else:
        arraytype = vtk.VTK_FLOAT
    imvtk = numpy_to_vtk(num_array=imnp_flat, deep=True, array_type=arraytype)
    image = vtk.vtkImageData()
    image.SetDimensions(imnp.shape)
    image.SetSpacing(spacing)
    points = image.GetPointData()
    points.SetScalars(imvtk)
    return image


def numpy2mhd(imnp, spacing, filename, header=None):
    """writes a numpy array in metaImage (mhd+raw)"""
    # turns the numpy array to vtk array
    writer = vtk.vtkMetaImageWriter()
    try:
        writer.SetInputData(numpy2vtk(imnp, spacing))
    except:
        writer.SetInput(numpy2vtk(imnp, spacing))
    # writes it as a mhd+raw format
    writer.SetFileName(filename)
    writer.Write()
    # writer AIM header if provided
    if header is not None:
        with open(filename, "a") as f:
            f.write(
                """
!-------------------------------------------------------------------------------
                               AIM Log
!-------------------------------------------------------------------------------"""
            )
            for line in header:
                f.write(line + "\n")


def ext(filename, new_ext):
    """changes the file extension"""
    return filename.replace("." + filename.split(".")[-1], new_ext)


def mhd2stl(mhddir, stldir, cort_mask):
    """
    Imports mhd+zraw files and converts them into stl file readable by GMSH
    Author: Simone Poncioni
    Institute: MSB group, ARTORG Bern
    Date: 11.05.2022
    """
    # Load and read-in the files (MHD + ZRAW)
    path_mhd = str(Path(mhddir / str(cort_mask)).resolve())
    cort_mask_n = Path(cort_mask).name
    stl_name = ext(cort_mask_n, '.stl')
    reader = vtk.vtkMetaImageReader()
    reader.SetFileName(path_mhd)
    reader.Update()
    
    # Use vtkImageThreshold to remove the empty layer around the bone
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(reader.GetOutputPort())
    threshold.ThresholdByLower(0.25)  # remove empty layer
    threshold.ReplaceInOn()
    threshold.SetInValue(0)  # set all values below 1 to 0
    threshold.ReplaceOutOn()
    threshold.SetOutValue(1)  # set all values above 1 to 1
    threshold.Update()

    # Use vtkDiscreteMarchingCubes class to extract the surface
    print('...\tExtracting the surfaces')
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(threshold.GetOutputPort())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()

    # Read-in meta-data:
    # Load dimensions using `GetDataExtent`
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    # Save the extracted surface as an .stl file
    print(' ... Converting MHD to STL')
    writer = vtk.vtkSTLWriter()
    # writer.SetFileTypeToBinary()
    writer.SetFileTypeToASCII()  # TODO: substitute with previous line if not needed by Abaqus!
    writer.SetInputConnection(dmc.GetOutputPort())
    writer.SetFileName(Path(stldir / stl_name))
    writer.Write()
    return str(Path(stldir / stl_name).resolve())


# not apply the fix by considering vertical axis
def fill_contours_fixed(arr):
    return np.maximum.accumulate(arr, 1) &\
           np.maximum.accumulate(arr[:, ::-1], 1)[:, ::-1] &\
           np.maximum.accumulate(arr[::-1, :], 0)[::-1, :] &\
           np.maximum.accumulate(arr, 0)


def mhd2itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename, sitk.sitkUInt8)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    # superior nodes
    superior_voxel_np = np.copy(ct_scan)
    superior_voxel_np[0:superior_voxel_np.shape[0]-1, 0:superior_voxel_np.shape[1], 0:superior_voxel_np.shape[2]] = 0

    # inferior nodes
    inferior_voxel_np = np.copy(ct_scan)
    inferior_voxel_np[1:inferior_voxel_np.shape[0], 0:inferior_voxel_np.shape[1], 0:inferior_voxel_np.shape[2]] = 0

    # concatenate upper+inferior to full mask
    closed_mask_upper = np.stack([*ct_scan[:-1, :, :], superior_voxel_np[-1, :, :]])
    closed_mask = np.stack((inferior_voxel_np[-1, :, :], *closed_mask_upper[1:, :, :]))

    return closed_mask, spacing


def gmsh_repair_mesh(stl_name, gmsh_name, decimation_factor=10e4, max_elm_size=2.0, mesher_id_alg=10):
    """
    - Imports closed surface in STL format
    - Repairs it if it's not watertight
    - Meshes with gmsh the repaired closed surface into a volume in .msh format

    Args:
        stl_name (str): path of the .stl file
        gmsh_name (str): path of the gmsh file to write
        decimation_factor (int): magnitude of the quadratic decimation
        max_elm_size (float or None): Maximum length of an element in the volume mesh
        mesher_id (int): 3D unstructured algorithms: 1: Delaunay, 4: Frontal, 7: MMG3D, 10: HXT

    Returns:
        None
    """

    # load into trimesh and decimate to 'decimation_factor' faces
    mesh = trimesh.load(stl_name)
    print(f'Original mesh watertight? {mesh.is_watertight}')
    mesh_decimation = mesh.simplify_quadratic_decimation(decimation_factor)
    print(f'Decimated mesh watertight? {mesh_decimation.is_watertight}')

    if mesh_decimation.is_watertight is not True:
        print('Repairing surface to make it watertight')
        meshfix = pymeshfix.MeshFix(mesh_decimation.vertices, mesh_decimation.faces)
        meshfix.repair(verbose=True)
        mesh_repair = trimesh.Trimesh(vertices=meshfix.v, faces=meshfix.f)
        print(f'meshfix is watertight? {mesh_repair.is_watertight}')
    else:
        print('No need to repair')
        mesh_repair = mesh_decimation
    mesh_repair.export(ext(gmsh_name, '.stl')) # export .stl for cubit
    trimesh.interfaces.gmsh.to_volume(mesh_repair, file_name=gmsh_name, max_element=max_elm_size, mesher_id=mesher_id_alg)
    return None


def main(base_path, filename_sample):
    #########
    # TODO: substitute this with original from pipeline
    bone = {}
    bone['Spacing'] = np.array([0.0607, 0.0607, 0.0607])
    spacing = bone["Spacing"]
    # name = '/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/00_ORIGAIM/C0002231/C0002231_CORT_MASK.AIM;1'
    name = '/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/00_ORIGAIM/C0002231/C0002231_TRAB_MASK.AIM;1'

    #########
    origaim_path = Path(base_path, '00_ORIGAIM')
    filename_s = Path(origaim_path, filename_sample)
    
    
    masks = []
    masks_name = []
    for file in Path(filename_s).glob('*[CORT]*[TRAB]_MASK*'):
        masks.append(file)
        masks_name.append(file.name)
    print(masks_name)
    
    '''
    matching = [s for s in masks_name if any(xs in s for xs in masks_name)]
    print(matching)
    name_to_check = ['CORT', 'TRAB']

    print([url_string for extension in name_to_check if(extension in url_string)])
    
    if '*CORT_MASK*' not in masks and '*TRAB_MASK*' not in masks:
        raise ValueError('Could not find cort mask and/or trab mask, exiting')
    '''
    mhddir = Path(base_path, '01_AIM')
    stldir = Path(base_path, '03_MESH')
    stl_name = filename_sample + '.stl'

    for mask in masks_name:
        print(f'Mask being currently processed:\t{mask}')
        mask_cap = ext(mask, '') + '_cap02.mhd'
        gmsh_name = str(Path(stldir, ext(mask, '.msh')).resolve())
        # Convert original AIMs to numpy array
        bone, imvtk = read_aim(str(Path(filename_s, mask).resolve()), bone) #TODO: name
        print(" ... Converting AIM to MHD + ZRAW")
        vtk2mhd(imvtk, bone['Spacing'], str(Path(mhddir, ext(masks_name[0], '.mhd')).resolve()), header=None)
        closed_mask, spacing = mhd2itk(str(Path(mhddir, ext(masks_name[0], '.mhd')).resolve()))
        numpy2mhd(closed_mask, spacing, str(Path(mhddir, mask_cap).resolve()), header=None)
        
        # stl_name = mhd2stl(mhddir, stldir, mask_cap)
        # gmsh_repair_mesh(str(Path(stldir, stl_name).resolve()), gmsh_name, decimation_factor = 10e5, max_elm_size = 2.0, mesher_id_alg=1)


if __name__ == "__main__":

    base_path_m = r'/home/simoneponcioni/Documents/01_PHD/03_Methods/Meshing/Meshing/'
    filename_sample_m = 'C0001406'
    
    main(base_path=base_path_m, filename_sample=filename_sample_m)
