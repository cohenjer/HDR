# Time for a little dance with matlab and python dataset formats
import scipy.io
import xarray
import tensorly as tl
import numpy as np

def load_eem():
    """
    Load the EEM dataset from a .mat file and return it as a tensor.
    The dataset is normalized and noise is added for demonstration purposes.
    """
    # Load the EEM data
    data = scipy.io.loadmat('../../tensorly_hdr/dataset/eem.mat')
    tensor = data['X']['data']
    tensor = tl.tensor(tensor[0,0])
    tensor = tensor/tl.max(tensor)
    # adding noise
    tensor_noised = tensor + 0.2*np.random.randn(*tensor.shape)
    # Creating a dataset for tlviz
    dataset = xarray.DataArray(
        data = tensor_noised,
        coords={
            "Sample index": np.linspace(1,18,18),
            "Emission wavelength": np.linspace(250, 500, 251),
            "Excitation wavelength": np.linspace(210, 310, 21)
        },
        dims = ["Sample index", "Emission wavelength", "Excitation wavelength"]
    )
    return dataset, tensor_noised, tensor