"""The data structure module holds model classes."""
import enum

import futils.conversion as conversion
import SimpleITK as sitk


class BoneImageTypes(enum.Enum):
    """Represents human readable image types."""
    CortMask = 1  #: The SCANCO cortical mask.
    TrabMask = 2  #: The SCANCO trabecular mask.
    RegistrationTransform = 3  #: The registration transformation


class BoneImage:
    """Represents a bone image."""
    
    def __init__(self, id_: str, path: str, images: dict, transformation: sitk.Transform):
        """Initializes a new instance of the BoneImage class.

        Args:
            id_ (str): An identifier.
            path (str): Full path to the image directory.
            images (dict): The images, where the key is a :py:class:`BoneImageTypes` and the value is a
            SimpleITK image.
        """

        self.id_ = id_
        self.path = path
        self.images = images
        self.transformation = transformation

        # ensure we have an image to get the image properties
        if len(images) == 0:
            raise ValueError('No images provided')

        self.image_properties = conversion.ImageProperties(self.images[list(self.images.keys())[0]])
        self.feature_images = {}
        self.feature_matrix = None  # a tuple (features, labels),
        # where the shape of features is (n, number_of_features) and the shape of labels is (n, 1)
        # with n being the amount of voxels
