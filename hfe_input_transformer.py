# just a few functions to prepare the input data in the same way as in the pipeline
import SimpleITK as sitk
import numpy as np


def numpy_reader_to_sitk(path_np: str, spacing: list = [0.0607, 0.0607, 0.0607]):
    """
    Little helper to read numpy array and convert it to sitk image
    Used to get same import conditions as pipeline, where 'dict(bone)' contains a np array of the cortical mask
    """
    cort_mask_np = np.load(path_np)

    sitk_image = sitk.GetImageFromArray(cort_mask_np)
    sitk_image = sitk.PermuteAxes(sitk_image, [2, 1, 0])
    sitk_image = sitk.Flip(sitk_image, [False, True, False])
    sitk_image.SetSpacing(spacing)
    return sitk_image


def pad_image(image, iso_pad_size: int):
    """
    Pads the input image with a constant value (background value) to increase its size.
    Padding is used to prevent having contours on the edges of the image,
    which would cause the spline fitting to fail.
    Padding is performed on the transverse plane only
    (image orientation is assumed to be z, y, x)

    Args:
        image (SimpleITK.Image): The input image to be padded.
        iso_pad_size (int): The size of the padding to be added to each dimension.

    Returns:
        SimpleITK.Image: The padded image.
    """
    constant = int(sitk.GetArrayFromImage(image).min())
    image_pad = sitk.ConstantPad(
        image,
        (iso_pad_size, iso_pad_size, 0),
        (iso_pad_size, iso_pad_size, 0),
        constant,
    )
    return image_pad


def hfe_input(path_np_s: str):
    """
    # These few lines are only necessary if you are testing HFE pipeline
    # If you have a CORT_MASK.mhd in img_settings['img_basefilename'] you can skip this part
    """
    sitk_image = numpy_reader_to_sitk(
        path_np=path_np_s,
        spacing=[0.0607, 0.0607, 0.0607],
    )

    # add padding to sitk image
    sitk_padded = pad_image(sitk_image, iso_pad_size=10)

    # save image to mhd format for debugging
    sitk.WriteImage(
        sitk_padded,
        "99_testing_prototyping/pipeline_implementation_errors/C0001406_CORTMASK.mhd",
    )

    return sitk_padded
