import io


import pickle

# First, import the module
from spikeinterface.core.base import BaseExtractor  # assuming from_dict is in BaseExtractor

# Store original function
original_from_dict = BaseExtractor.from_dict


# Define debug wrapper
def debug_from_dict(*args, **kwargs):
    print(f"from_dict called with args: {args}")
    print(f"kwargs: {kwargs}")
    result = original_from_dict(*args, **kwargs)
    print(f"Result type: {type(result)}")
    return result


# Apply the monkey patch
BaseExtractor.from_dict = debug_from_dict


def hex_2_rgb(hex_color):  # Orange color
    rgb_color = tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (1, 3, 5))
    return rgb_color


def pickle_this(thing_to_pickle, file_name):
    """
    Pickles things
    Args (2):
        thing_to_pickle: anything you want to pickle
        file_name: str, filename that ends with .pkl
    Returns:
        none
    """
    with open(file_name, "wb") as file:
        pickle.dump(thing_to_pickle, file)


def unpickle_this(pickle_file):
    """
    Unpickles things
    Args (1):
        file_name: str, pickle filename that already exists and ends with .pkl
    Returns:
        pickled item
    """
    with open(pickle_file, "rb") as file:
        return pickle.load(file)


class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "LFP_collection":
            renamed_module = "lfp.lfp_analysis.LFP_collection"
        if module == "LFP_recording":
            renamed_module = "lfp.lfp_analysis.LFP_recording"
        if module == "spikeinterface.core.base":
            pass

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)


with open(
    r"C:\Users\megha\UFL Dropbox\Meghan Cum\Padilla-Coreano Lab\2024\Cum_SocialMemEphys_pilot2\processed_lfp_pickles\novel_p1_processed_lfp.pkl",
    "rb",
) as file:
    nov_lfp = renamed_load(file)
