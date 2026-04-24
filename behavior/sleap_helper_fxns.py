import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def print_hdf5_info(obj, indent=0):
    """Recursively prints information about datasets and groups in an HDF5 file."""
    if isinstance(obj, h5py.File) or isinstance(obj, h5py.Group):
        for key in obj.keys():
            print(' ' * indent + f"{'/' if indent == 0 else ''}{key}:")
            print_hdf5_info(obj[key], indent + 4)
    elif isinstance(obj, h5py.Dataset):
        print(' ' * indent + f"{'/' if indent == 0 else ''}{obj.name} (Dataset)")
        print(' ' * (indent + 4) + f"Shape: {obj.shape}")
        print(' ' * (indent + 4) + f"Dtype: {obj.dtype}")
        if obj.attrs:
            print(' ' * (indent + 4) + "Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(' ' * (indent + 8) + f"{attr_name}: {attr_value}")
    else:
        print(' ' * indent + f"Unknown object: {obj}")

def make_dict(my_list):
    my_dict = {}
    for i in range(len(my_list)):
        my_dict[my_list[i]] = i
    return my_dict

class sleap_vid():
    """
    A class for sleap videos initiated by the filename (.h5 only) and the number of mice 
    you annotated. 

    Attributes:
        name: str, path to an .h5 file 
        mice: number of mice in the experiment, also called tracks 
        locations: np.array, a multidimensional array of dimensions
            (frames, nodes, 2 (x/y coordinates), track_no)
        tracks: list of str, names of tracks assigned during sleap annotation
        nodes: list of str, names of nodes assigned during sleap annotation
        track_dict: dict, keys: str, names of tracks, values: int, dimension of location
            associated with that track
        node_dict: dict, keys: str, names of nodes, values: int, dimension of location 
            associated with that node
                
    Methods:
        smooth_locations: smooths locations with a savtisky-golay filter
        node_velocity: calculates and returns the velocity of a given node 
            for all tracks
        distance_between_mice: calculates and returns distances between tracks
            given a node
        distances_between_nodes: calculates and returns distances between nodes within 
            one skeleton for all tracks
        distances_to_point: calculates and returns distances between a point (x,y coordinates) 
            and a node for each track
        node_angles: calculates and returns the angle between three nodes for each track
        point_angles: calculates and returns the angle between a point and two nodes for
            each track
        orientation: calculates and returns the orientation of the two tracks to each other
        create_events_array: TBD
    """

    def __init__(self, filename):
        """
        Initiates a sleap_vid class instance 
        Args (2)
            filename: str, path to .h5 file 
            track_no: int, number of mice that you annotated for
        
        Returns: 
            a sleap_vid class instance 
        """
        self.name = '_'.join((filename.split('/')[-1].split('_'))[0:4])
        #self.mice = track_no
        self.__get_info__(filename)

    def __get_info__(self, filename):                                                                                                       
      with h5py.File(filename, "r") as f:  
          tracks = [n.decode() for n in f["track_names"][:]]
          locations = f["tracks"][:].T  # (frames, nodes, 2, n_tracks)
          nodes = [n.decode() for n in f["node_names"][:]]     
          point_scores = f['point_scores'][:]                                                                           
  
      #self.locations = fill_missing(locations)                                                                                            
      self.tracks = tracks                 
      self.nodes = nodes  
      self.locations = locations  
      self.point_scores = point_scores                                                                                                             
      self.track_dict = make_dict(tracks)
      self.node_dict = make_dict(nodes)                                                                                                   
                                           


def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

def smooth_diff(node_loc, deriv, win=25, poly=3):
    """
    node_loc is a [frames, 2] array
    
    win defines the window to smooth over
    
    poly defines the order of the polynomial
    to fit with
    
    """
    node_vel = np.zeros_like(node_loc)
    for c in range(node_loc.shape[-1]):
        node_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv)
    if deriv != 0:
        node_vel = np.linalg.norm(node_vel,axis=1)
    return node_vel


def smooth_locations(locations, win=25, poly=3):
    """
    Smooths locations using a savitsky-golay filter (fxn from numpy, code from
    sleap.ai) and reassigns self.locations to the smoothed locations

    Args (0 required, 2 total):
        win: int, length of filter window
        poly: int, the order of the polynomial used to fit the samples

    Returns:
        none
    """
    for node in range(locations.shape[1]):
        for track in range(locations.shape[-1]):
            nodeloc = locations[:,node,:,track]
            smoothed_node = smooth_diff(nodeloc, deriv=0, win=win, poly=poly)
            locations[:,node,:,track] = smoothed_node
    return locations
    

def node_velocity(locations, node_idx, win=25, poly=3, normalization_factor=None):
    """
    takes in node and returns the velocity of that node 
    for each mouse

    Args: 
        node: string, name of node
    
    Returns:
        velocities: 2d np.array of floats (d = 2 x #of frames)
            where each element is the velocity for that node
            distances[0] = velocities for mouse2
            distances[1] = velocities for mouse2
    """
    node_loc1 = locations[:, node_idx, :, 0]
    node_loc2 = locations[:, node_idx, :, 1]
    if normalization_factor is not None:
        node_loc1 = node_loc1 * normalization_factor
        node_loc2 = node_loc2 * normalization_factor
    m1_vel = smooth_diff(node_loc1, deriv = 1, win=win, poly=poly)
    m2_vel = smooth_diff(node_loc2, deriv = 1, win=win, poly=poly)
    velocities = np.array([m1_vel,m2_vel])
    return velocities 

def distances_between_mice(locations, node_index, normalization_factor=None):
    """
    takes in node name
    returns a list of distances between the nodes of the two mice

    Args:
        node: string, name of node
    Returns:
        c_list: 1D np.array of floats (d = # of frames)
    """
    
    x1 = locations[:,node_index,0,0]
    y1 = locations[:,node_index,1,0]
    # x , y coordinate of nose for mouse 1
    x2 = locations[:,node_index,0,1]
    y2 =  locations[:,node_index,1,1]
    # x and y coordinate of nose of mouse 2
    # solve for c using pythagroean theory
    distances = np.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))
    if normalization_factor is not None: 
        distances = distances * normalization_factor
        distances = distances.T.flatten()
    return distances

def distances_between_nodes(locations, node_index1, node_index2, normalization_factor = None):
    """
    takes in two nodes and returns the distances between those nodes 
    for each mouse

    Args: 
        node1: string, name of node 1
        node2: string, name of node 2 
    
    Returns:
        distances: 2d np.array of floats (d = 2 x #of frames)
            where each element is the distance between node1 and node2 
            distances[0] = distances for mouse1
            distances[1] = distances for mouse2
    """
    x1,y1 = locations[:, node_index1,0,0], locations[:,node_index1,1,0]
    # x , y coordinate of node 1 for mouse 1
    x2,y2 = locations[:,node_index2,0,0], locations[:,node_index2,1,0]
    # x, y coordiantes of node 2 for mouse 1
    x3, y3 = locations[:,node_index1,0,1], locations[:,node_index1,1,1]
    # x and y coordinate of node 1 of mouse 2
    x4, y4 = locations[:,node_index2,0,1], locations[:,node_index2,1,1]
    # solve for c using pythagroean theory
    c2 = np.sqrt(((x3 -x4)**2)+ ((y3 - y4)**2))
    c1 = np.sqrt(((x1 - x2)**2) + ((y1 - y2)**2))
    if normalization_factor is not None:
        c2 = (c2*normalization_factor).T.flatten()
        c1 = (c1*normalization_factor).T.flatten()
    distances = np.array([c1, c2])
    return distances

def node_angles(locations, node1_index, node2_index, node3_index):
    """
    takes in locations and three nodes, calculates angle between the three points 
    with the second node being the center point
    
    Args:  
        node1: string, name of node 1
        node2: string, name of node 2 
        node3: string, name of node 3

    Returns:
        ang: 2d np. array (d = 2 x # of frames)
            where each element is the angle between 
            node1 and node3 with node2 as center point 
            angles_all_mice[0] = angles for mouse1
            angles_all_mice[1] = angles for mouse2
    """
    ax = locations[:,node1_index, 0, :]
    ay = locations[:,node1_index, 1, :]
    bx = locations[:,node2_index, 0, :]
    by = locations[:,node2_index, 1, :]
    cx = locations[:,node3_index,0,:]
    cy = locations[:,node3_index, 1, :]
    ang = np.arctan2(cy-by, cx-bx) - np.arctan2(ay-by, ax-bx) 
    ang_swapped = np.arctan2(cy-by, cx-bx) - np.arctan2(cy-by, cx-bx) 
    ang = np.maximum(ang, ang_swapped)
    return ang.T

def orientation(locations, nose_node, head_node, other_node):
    """
    Takes in locations and nose and head node index to calculate the angle of orientation from one mouse
    facing whatever other_node is chosen
    between mice where two mice facing each other results in pi
    theta = 0 means they are not facing each other 
    """
   
    ax = locations[:, nose_node, 0, 0]
    ay = locations[:, nose_node, 1, 0]
    bx = locations[:,head_node, 0, 0]
    by = locations[:,head_node, 1, 0]
    cx = locations[:, other_node, 0, 1]
    cy = locations[:,other_node, 1, 1]
    ang_m1 = np.arctan2(cy-by, cx-bx) - np.arctan2(ay-by, ax-bx) 
    ang_m1_swapped = np.arctan2(ay-by, ax-bx) - np.arctan2(cy-by, cx-bx)
    ax = locations[:, nose_node, 0, 1]
    ay = locations[:, nose_node, 1, 1]
    bx = locations[:,head_node, 0, 1]
    by = locations[:,head_node, 1, 1]
    cx = locations[:, other_node, 0, 0]
    cy = locations[:,other_node, 1, 0]
    ang_m2 = np.arctan2(cy-by, cx-bx) - np.arctan2(ay-by, ax-bx) 
    ang_m2_swapped = np.arctan2(ay-by, ax-bx) - np.arctan2(cy-by, cx-bx) 
    ang_m1 = np.maximum(ang_m1,ang_m1_swapped)
    ang_m2 = np.maximum(ang_m2,ang_m2_swapped)
    return np.array([ang_m1, ang_m2])

def compute_moving_angle(locations, node_idx):
    """
    Compute velocity and moving angle for each point in a trajectory.
    
    Parameters:
    - trajectory: np.array of shape (N, 2), where N is the number of frames, and 2 represents (x, y).

    Returns:
    - velocity: np.array of shape (N,) containing velocity magnitudes.
    - moving_angle: np.array of shape (N,) containing movement direction angles in degrees.
    """
    moving_angle1 = np.zeros(locations.shape[0])
    moving_angle2 = np.zeros(locations.shape[0])
    trajectories1 = locations[:, node_idx, :, 0]
    trajectories2 = locations[:, node_idx, :, 1]
    # Compute velocity as the difference between the next and previous points
    diff1_vectors = trajectories1[1:] - trajectories1[:-1]
    diff2_vectors = trajectories2[1:] - trajectories2[:-1]

    # Compute moving angle using arctan2 (y component first, then x)
    moving_angle1[:-1] = np.degrees(np.arctan2(diff1_vectors[:, 1], diff1_vectors[:, 0]))
    moving_angle2[:-1] = np.degrees(np.arctan2(diff2_vectors[:, 1], diff2_vectors[:, 0]))

    moving_angle = np.stack([moving_angle1, moving_angle2])

    return moving_angle   
                                                                            
def get_ins_outs(boris_df, fps):       
    if len(boris_df[boris_df['Behavior'] == 'mice back in']["Start (s)"]) == 6:
        if 'Image index start' not in boris_df.columns:
            ins_times = np.array(boris_df[boris_df['Behavior'] == 'mice back in']["Start (s)"])[[0,2,4]]
            outs_times = np.array(boris_df[boris_df['Behavior'] == 'mice taken out']["Start (s)"])[[0,2]]
            ins_frames = np.round(ins_times * fps).astype(int)
            outs_frames = np.round(outs_times * fps).astype(int)
        else:
            ins_frames = np.array(boris_df[boris_df['Behavior'] == 'mice back in']["Image index start"])[[0,2,4]]
            outs_frames= np.array(boris_df[boris_df['Behavior'] == 'mice taken out']["Image index start"])[[0,2]]
    else:
        if 'Image index start' not in boris_df.columns:
            ins_times = np.array(boris_df[boris_df['Behavior'] == 'mice back in']["Start (s)"])
            outs_times = np.array(boris_df[boris_df['Behavior'] == 'mice taken out']["Start (s)"])
            ins_frames = np.round(ins_times * fps).astype(int)
            outs_frames = np.round(outs_times * fps).astype(int)
            
        else:
            ins_frames = np.array(boris_df[boris_df['Behavior'] == 'mice back in']["Image index start"])
            outs_frames = np.array(boris_df[boris_df['Behavior'] == 'mice taken out']["Image index start"])
 
    return ins_frames, outs_frames
 
def sleap_to_1ms(sleap_array, timestamps_ms, start_ms=None, stop_ms=None):
    """
    Resample (frames, nodes, axes) into 1ms bins by forward-filling.

    Parameters
    ----------
    sleap_dict    : 'locations': [n_frames, nodes, axes]
                    'first_timestamp': int
    timestamps_ms : (n_frames,) camera timestamps in ms, play indexed
    start_ms : first ms bin (default: floor of first timestamp), play indexed
    stop_ms   : last ms bin  (default: ceil of last timestamp), play indexed

    Returns
    -------
    dict:
        'locations_1ms' : (n_ms_bins, nodes, axes) — forward filled
        'frame_index'   : (n_ms_bins,) int — which original frame each bin came from
        'time_ms'       : (n_ms_bins,) int — ms index of each bin
    """
    if start_ms is None:
        start_ms = int(np.floor(timestamps_ms[0]))
    if stop_ms is None:
        stop_ms = int(np.ceil(timestamps_ms[-1]))

    n_bins    = stop_ms - start_ms + 1
    time_ms   = np.arange(start_ms, stop_ms + 1)  # (n_bins,)
    # for each ms bin, find the most recent frame (forward fill)
    # searchsorted with 'right' gives insertion point → subtract 1 = last frame before this ms
    frame_indices = np.searchsorted(timestamps_ms, time_ms, side='right') - 1
    frame_indices = np.clip(frame_indices, 0, sleap_array.shape[0] - 1)

    return {
        'locations_1ms' : sleap_array[frame_indices],   # (n_bins, nodes, axes)
        'frame_index'   : frame_indices,                 # (n_bins,) — which frame
        'time_ms'       : time_ms,                       # (n_bins,) — absolute ms
    }


