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


def load_gcms_interval():
    """
    Load the raw GCMS dataset from a .mat file and return it as a tensor.
    """
    # Load the GCMS data
    data = scipy.io.loadmat('../../tensorly_hdr/dataset/gcms_data.mat')
    tensor = tl.tensor(data['tensor'])
    time_steps = data['time_steps'][0]
    # Creating a dataset for tlviz
    dataset = xarray.DataArray(
        data=tensor,
        coords={
            "Mass over charge index": np.linspace(1,155,155),
            "Elution time": time_steps,
            "Sample index": np.linspace(1, 251, 251)
        },
        dims=["Mass over charge index", "Elution time", "Sample index"]
    )
    
    # To load the GCMS data, m/z spectra x time x experiment from the set of all intervals
    #data = loadmat('../../tensorly_hdr/dataset/Intervals.mat')
    #tensor = tl.tensor(data['Int38'])
    #tensor = tensor/tl.max(tensor)
    #time_steps = data["rt38"][0]
    ## Creating a dataset for tlviz
    #dataset = xarray.DataArray(
        #data = tensor,
        #coords={
            #"Mass over charge index": np.linspace(1,155,155),
            #"Elution time": time_steps,
            #"Sample index": np.linspace(1, 251, 251)
        #},
        #dims = ["Mass over charge index", "Elution time", "Sample index"]
    #)
    
    return dataset, tensor, time_steps


