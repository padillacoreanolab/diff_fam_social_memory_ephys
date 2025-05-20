import pandas as pd
import numpy as np
import lfp.trodes.read_exported as tr
import os

def rec_to_ecu(rec_to_ecu_dict, rec_path, unit = 'timestamp', sampling_rate=20000):
    '''
    This function takes in a single rec folder that contains merged.DIO folders within it and uses a rec to ecu mapping
    to create an event dictionary of inputs to timestamp arrays to be used in LFP or spike analyses. Resulting timestamp
    arrays are zero-ed to the onset of recording, NOT to the onset of streaming. 
    Rec folder MUST have DIO folders in it. All rec_path variables must be strings with an 'r' in front of it.
    ex. r'test_rec.rec/test_rec1_merged.rec'

    Args(2 required, 1 optional):
        rec_to_ecu_dict: dict, rec names to dictionaries of ecu din numbers to physical inputs
            keys: str, merged.rec file name exactly as they appear in the rec folder
            values: dict, ecu to event name dict,
                keys: str, 'dio_ECU_DinX' where X is the input number from the medpc system
                values: str, user defined name for what that input goes to ie port
        rec_path: path, this means a str of the path to the rec folder with an r in front of it
            example: r'C:/Users/experiment/data/test_rec.rec'
        unit: str, {'timestamp', 'ms'}, default='timestamp'. if 'timestamp' the data from the DIO
            files are not converted and kept at frequency of the system so resulting event_dicts are 
            in units of 'timestamps'. if unit = 'ms'. uses sampling_rate to convert event_dict start 
            and stop times into ms 
        sampling_rate: int, default=20000, sampling rate of system in Hz, used to convert from timestamps to 
            ms if specified

    Returns (1):
        event_dict: dict, inputs to timestamp arrays
            keys: user defined inputs from rec_to_ecu dict 
            values: nd.array, numpy array of shape [n,2] where n is the number of events and 2 is 
                start and stop times for each event. These are in units of timestamps at the Hz
                of the system (usually 20kHz for spikegadgets headstages) or ms depending on unit parameter
    '''
    event_dict = {}
    for folder in os.listdir(rec_path):
        dio_path = os.path.join(rec_path, folder)
        if os.path.isdir(dio_path) and folder.endswith('.DIO'):
            files = tr.organize_single_trodes_export((dio_path))
            merged_rec = list(files.values())[0]['original_file']
            event_dict[merged_rec] = {}
            for din in rec_to_ecu_dict[merged_rec].keys():
                din_dict = files[din]
                if len(din_dict['data']) == 1:
                    pass
                else:
                    array = din_dict['data'][1:]
                    first_timestamp = int(din_dict['first_timestamp'])
                    result = []
                    current_start = None
                    for timestamp, flag in array:
                        if flag == 1 and current_start is None:
                            # Found a start point
                            current_start = timestamp
                        elif flag == 0 and current_start is not None:
                            # Found an end point - create a pair and reset
                            result.append([current_start, timestamp])
                            current_start = None
                    result_array = np.array(result)
                    result_array = result_array - first_timestamp
                    if unit == 'ms':
                        result_array = result_array / sampling_rate * 1000   
                    event_dict[merged_rec][rec_to_ecu_dict[merged_rec][din]] = np.array(result)
    return event_dict

def merged_rec_to_box(box_to_ecu_dict, rec_to_box_dict, data_path, unit = 'timestamps', sampling_rate = 20000):
    """
    This function is recommended for batch processing multiple recordings across many rec folders. It assumes
    subjects have stayed in the same box for the whole recording ie each mergec.rec file happend in a single box.  

    It takes in data path to multiple rec folders containing merged.DIO folders within them and uses
    two dictionaries to map ecu inputs to boxes and rec name to boxes rec to ecu to create an event dictionary 
    of inputs to timestamp arrays to be used in LFP or spike analyses. Resulting timestamp
    arrays are zero-ed to the onset of recording, NOT to the onset of streaming.  
    Rec folders MUST have DIO folders in them. All data_path variables must be strings with an 'r' in front of it.
    ex. r'data/test_rec.rec/test_rec1_merged.rec'

    Args(3 required, 2 optional):
        box_to_ecu_dict: dic,t box number to dictionaries of Ecu mappings
            keys: int, box number as it appears in the rec_to_box_dict
            values: dict, ecu to event name dict,
                keys: str, 'dio_ECU_DinX' where X is the input number from the medpc system
                values: str, user defined name for what that input goes to ie port
        rec_to_box_dict: dict, rec names to box number
            keys: str, merged.rec file name exactly as they appear in the rec folder
            values: int, box number 
        data_path: path, this means a str of the path to the data folder with an r in front of it
            example: r'C:/Users/experiment/data'
        unit: str, {'timestamp', 'ms'}, default='timestamp'. if 'timestamp' the data from the DIO
            files are not converted and kept at frequency of the system so resulting event_dicts are 
            in units of 'timestamps'. if unit = 'ms'. uses sampling_rate to convert event_dict start 
            and stop times into ms 
        sampling_rate: int, default=20000, sampling rate of system in Hz, used to convert from timestamps to 
            ms if specified

    Returns (1):
        event_dict: dict, inputs to timestamp arrays
            keys: user defined inputs from rec_to_ecu dict 
            values: nd.array, numpy array of shape [n,2] where n is the number of events and 2 is 
                start and stop times for each event. These are in units of timestamps at the Hz
                of the system (usually 20kHz for spikegadgets headstages) or ms depending on unit parameter
    """
    event_dict = {}
    for rec in os.listdir(data_path):
        # Construct the full path
        rec_path = os.path.join(data_path, rec)
        # Check if it's a directory and ends with .rec
        if os.path.isdir(rec_path) and rec.endswith('.rec'):
            for folder in os.listdir(rec_path):
                dio_path = os.path.join(rec_path, folder)
                if os.path.isdir(dio_path) and folder.endswith('.DIO'):
                    files = tr.organize_single_trodes_export((dio_path))
                    merged_rec = list(files.values())[0]['original_file']
                    event_dict[merged_rec] = {}
                    box = rec_to_box_dict[merged_rec]
                    for din in box_to_ecu_dict[box].keys():
                        din_dict = files[din]
                        if len(din_dict['data']) == 1:
                            pass
                        else:
                            array = din_dict['data'][1:]
                            first_timestamp = int(din_dict['first_timestamp'])
                            result = []
                            current_start = None
                            for timestamp, flag in array:
                                if flag == 1 and current_start is None:
                                    # Found a start point
                                    current_start = timestamp
                                elif flag == 0 and current_start is not None:
                                    # Found an end point - create a pair and reset
                                    result.append([current_start, timestamp])
                                    current_start = None
                            result_array = np.array(result)
                            result_array = result_array - first_timestamp
                            if unit == 'ms':
                                result_array = result_array / sampling_rate * 1000 
                            event_dict[merged_rec][box_to_ecu_dict[box][din]] = np.array(result)
    return event_dict


#duplicated data with box numbwer for the potential of subjects switching boxes mid recording

def rec_to_box(box_to_ecu_dict, data_path, unit = 'timestamps', sampling_rate = 20000):
    """
    This function to recommended for experiments where subjects switch boxes during a recording. User will
    have  
    
    It takes in a data path to multiple rec folders containing merged.DIO folders within them and uses
    one input dictionary to map ecu inputs for each box to create an event dictionary 
    of inputs to timestamp arrays to be used in LFP or spike analyses. Resulting timestamp
    arrays are zero-ed to the onset of recording, NOT to the onset of streaming. The resulting event_dict is per
    rec folder (not per merged.rec recording) and hsa boxes to ECU inputs to event start and stops.

    Rec folders MUST have DIO folders in them. All data_path variables must be strings with an 'r' in front of it.
    ex. r'data/test_rec.rec/test_rec1_merged.rec'

    Args(2 required, 2 optional):
        box_to_ecu_dict: dic,t box number to dictionaries of Ecu mappings
            keys: int, box number as it appears in the rec_to_box_dict
            values: dict, ecu to event name dict,
                keys: str, 'dio_ECU_DinX' where X is the input number from the medpc system
                values: str, user defined name for what that input goes to ie port
        data_path: path, this means a str of the path to the data folder with an r in front of it
            example: r'C:/Users/experiment/data'
        unit: str, {'timestamp', 'ms'}, default='timestamp'. if 'timestamp' the data from the DIO
            files are not converted and kept at frequency of the system so resulting event_dicts are 
            in units of 'timestamps'. if unit = 'ms'. uses sampling_rate to convert event_dict start 
            and stop times into ms 
        sampling_rate: int, default=20000, sampling rate of system in Hz, used to convert from timestamps to 
            ms if specified

    Returns (1):
        event_dict: dict, inputs to timestamp arrays
            keys: user defined inputs from rec_to_ecu dict 
            values: nd.array, numpy array of shape [n,2] where n is the number of events and 2 is 
                start and stop times for each event. These are in units of timestamps at the Hz
                of the system (usually 20kHz for spikegadgets headstages) or ms depending on unit parameter
    """
    event_dict = {}
    for rec in os.listdir(data_path):
        # Construct the full path
        rec_path = os.path.join(data_path, rec)
        is_first = True
        event_dict[rec] = {}
        # Check if it's a directory and ends with .rec
        if os.path.isdir(rec_path) and rec.endswith('.rec') and is_first:
            for folder in os.listdir(rec_path):
                    dio_path = os.path.join(rec_path, folder)
                    if os.path.isdir(dio_path) and folder.endswith('.DIO'):
                        files = tr.organize_single_trodes_export((dio_path))
                        box_mapping = {}
                        for box, box_dict in box_to_ecu_dict.items():
                            box_input = {} 
                            for din in box_dict.keys():
                                din_dict = files[din] 
                                if len(din_dict['data']) == 1:
                                    
                                    pass
                                else:
                                    array = din_dict['data'][1:]
                                    result = []
                                    current_start = None
                                    for timestamp, flag in array:
                                        if flag == 1 and current_start is None:
                                            # Found a start point
                                            current_start = timestamp
                                        elif flag == 0 and current_start is not None:
                                            # Found an end point - create a pair and reset
                                            if unit == 'ms':
                                                current_start = current_start / sampling_rate * 1000
                                                timestamp = timestamp / sampling_rate * 1000
                                            result.append([current_start, timestamp])
                                            current_start = None
                                    box_input[box_to_ecu_dict[box][din]] = result
                                box_mapping[box] = box_input
                            event_dict[rec] = box_mapping
            is_first = False
        return event_dict

