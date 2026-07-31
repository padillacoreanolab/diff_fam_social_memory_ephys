import pandas as pd
import math
import os
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
import xml.etree.ElementTree as ET


def creating_new_df(individuals_array):
    columns = ["ID"]
    # Using a loop to add the suffix to each string
    # building columns
    suffix1 = "w"
    suffix2 = "m"
    w_array = []
    m_array = []
    for w in individuals_array:
        w_array.append(w + suffix1)
    for m in individuals_array:
        m_array.append(m + suffix2)
    columns.extend(w_array)
    columns.extend(m_array)
    calculations = ["w", "l", "w2", "l2", "DS"]
    columns.extend(calculations)
    # entering base values into the new data frame
    empty_dataframe = pd.DataFrame(columns=columns)
    empty_dataframe["ID"] = individuals_array
    # fill in the data frame with zeros
    pd.set_option("future.no_silent_downcasting", True)
    new_dataframe = empty_dataframe.fillna(0).infer_objects(copy=False)
    new_dataframe[calculations] = new_dataframe[calculations].astype(float)
    return new_dataframe, w_array, m_array


# solving for all values: w, l, w2, l2, and DS
def solving_for_ds(df, individuals_array, dataframe, w_array, m_array):
    for i in range(len(individuals_array)):
        individual1 = individuals_array[i]
        for j in w_array:
            w_columns = j
            individual2 = j.rstrip("w")
            if individual1 != individual2:
                win = (df["winner"] == float(individual1)) & (
                    df["loser"] == float(individual2)
                ).astype(int)
                number_of_wins = df[win].shape[0]
                dataframe.at[i, w_columns] = number_of_wins
        for k in m_array:
            m_columns = k
            individual2 = k.rstrip("m")
            if individual1 != individual2:
                win = (
                    (
                        (df["winner"] == float(individual1))
                        & (df["loser"] == float(individual2))
                    )
                    .astype(int)
                    .sum()
                )
                lose = (
                    (
                        (df["winner"] == float(individual2))
                        & (df["loser"] == float(individual1))
                    )
                    .astype(int)
                    .sum()
                )
                matches = win + lose
                dataframe.at[i, m_columns] = matches
    for i in range(len(individuals_array)):
        # w = sum of P(ij) win-rate    P(ij)=(individual1/opponent)
        w = 0
        # l = sum of P(ji) loss-rate   P(ji)=1-(individual1/opponent)
        l = 0
        for j in range(len(individuals_array)):
            num_wins = dataframe.at[i, w_array[j]]
            num_matches = dataframe.at[i, m_array[j]]
            # ensure that there are matches between the two
            if math.isnan(num_matches) or num_matches == 0:
                continue
            if math.isnan(num_wins):
                num_wins = 0
            pw = num_wins / num_matches
            pl = l - pw
            w += pw
            l += pl
        dataframe.at[i, "w"] = float(w)
        dataframe.at[i, "l"] = float(l)

    # calculate the w2 and l2 values
    for i in range(len(individuals_array)):
        # w2 = sum of all P(ij) * w(j)
        w2 = 0
        # l2 = sum of all (1-P(ij)) * l(j)
        l2 = 0
        for j in range(len(individuals_array)):
            num_wins = dataframe.at[i, w_array[j]]
            num_matches = dataframe.at[i, m_array[j]]
            # ensure that there are matches between the two
            if math.isnan(num_matches) or num_matches == 0:
                continue
            if math.isnan(num_wins):
                num_wins = 0
            pw = num_wins / num_matches
            pl = 1 - pw
            w_opponent = dataframe.at[j, "w"]
            l_opponent = dataframe.at[j, "l"]
            w2 += pw * w_opponent
            l2 += pl * l_opponent
        dataframe.at[i, "w2"] = float(w2)
        dataframe.at[i, "l2"] = float(l2)

    # calculating David's score = w + w2 - l - l2
    for i in range(len(individuals_array)):
        w = dataframe.at[i, "w"]
        l = dataframe.at[i, "l"]
        w2 = dataframe.at[i, "w2"]
        l2 = dataframe.at[i, "l2"]
        ds = w + w2 - l - l2
        final_ds = round(ds, 3)
        dataframe.at[i, "DS"] = float(final_ds)
    data_to_keep = ["ID", "DS"]
    dataframe = dataframe[data_to_keep]
    return dataframe


def urine_tube(df):
    # Format column names
    df = df.copy()
    df.columns = [str(col).lower().strip().replace(" ", "_") for col in df.columns]
    """For some reason there are two loser columns under the tube file so..."""
    if "loser1" in df.columns:
        replace_values = {
            col: 0 if pd.api.types.is_numeric_dtype(df[col]) else pd.NaT
            for col in df.columns
        }
        df.fillna(value=replace_values, inplace=True)
        df["loser"] = df["loser1"] + df["loser2"]

    # Remove unneeded columns
    to_keep = ["winner", "loser"]
    df = df[to_keep]
    df["winner"] = df["winner"].astype(str).str.strip()
    df["loser"] = df["loser"].astype(str).str.strip()
    df["winner"] = df["winner"].astype(float)
    df["loser"] = df["loser"].astype(float)

    # insure only valid mouse id'd mice are allowed
    df.loc[df["winner"] == 0.0, "winner"] = np.nan
    df = df.dropna(subset=["winner"]).copy()

    # identifies all unique individuals in the data set
    all_individuals = np.array(pd.unique(df[["winner", "loser"]].values.ravel()))

    # sorts individual IDs
    sorted_values = np.sort(all_individuals)
    all_individuals_array = sorted_values.astype(str)
    if df["winner"].isna().any() is True:
        return None

    # creating a dataframe using function.
    new_dataframe, w_array, m_array = creating_new_df(all_individuals_array)
    """new data frame has columns with the wins of each mouse against another and the total matchups,
	it also has columns for mouse ID's, david score, and w, l, w2, l2"""
    final_df = solving_for_ds(
        df, all_individuals_array, new_dataframe, w_array, m_array
    )
    return final_df


# filling out the number of matches won by each mouse against another and how many matches they had.


def reward_comp(file):
    # Read the Excel file with header
    df = pd.read_excel(file, header=0)
    df.columns = [str(col).lower().strip().replace(" ", "_") for col in df.columns]
    # Remove unneeded columns
    to_keep = ["mouse_1_wins", "mouse_2_wins", "match"]
    # Keep only the specified columns
    df = df[to_keep]
    # splitting matches to find specific individuals
    individuals = set()
    df["match"] = df["match"].str.replace(" ", "")
    for match_str in df["match"]:
        if "vs" in match_str:
            individuals.update(match_str.split("vs"))
        else:
            individuals.update(match_str.split("v"))
    individuals_list = list(individuals)
    float_list = [float(value) for value in individuals_list]
    float_list.sort()
    all_individuals_array = [str(value) for value in float_list]

    # creating a dataframe using function.
    new_dataframe, w_array, m_array = creating_new_df(all_individuals_array)

    # finding the mice in each match and naming them as mouse 1 or 2 and finding all matches of these mice
    for n, row in df.iterrows():
        match = df.at[n, "match"]
        # splitting the match to find the mouse IDs
        if "vs" in match:
            split_match = match.split("vs")
        else:
            split_match = match.split("v")
        mouse1 = split_match[0]
        # locate mouse 1 location in all_individuals_array
        index_of_mouse1 = all_individuals_array.index(mouse1)
        mouse2 = split_match[1]
        # locate mouse 2 location in all_individuals_array
        index_of_mouse2 = all_individuals_array.index(mouse2)
        mouse1w = df.at[n, "mouse_1_wins"]
        mouse2w = df.at[n, "mouse_2_wins"]
        new_dataframe.loc[index_of_mouse1, mouse2 + "w"] += int(mouse1w)
        new_dataframe.loc[index_of_mouse2, mouse1 + "w"] += int(mouse2w)

        # total matches between the 2 mice
        mouse_m = mouse1w + mouse2w
        new_dataframe.at[index_of_mouse1, mouse2 + "m"] += int(mouse_m)
        new_dataframe.at[index_of_mouse2, mouse1 + "m"] += int(mouse_m)

    for i in range(len(all_individuals_array)):
        # w = sum of P(ij) win-rate    P(ij)=(individual1/opponent)
        w = 0
        # l = sum of P(ji) loss-rate   P(ji)=1-(individual1/opponent)
        l = 0
        for j in range(len(all_individuals_array)):
            num_wins = new_dataframe.at[i, w_array[j]]
            num_matches = new_dataframe.at[i, m_array[j]]
            # ensure that there are matches between the two
            if math.isnan(num_matches) or num_matches == 0:
                continue
            if math.isnan(num_wins):
                num_wins = 0
            pw = num_wins / num_matches
            pl = 1 - pw
            w += pw
            l += pl
        new_dataframe.at[i, "w"] = float(w)
        new_dataframe.at[i, "l"] = float(l)

    # calculate the w2 and l2 values
    for i in range(len(all_individuals_array)):
        # w2 = sum of all P(ij) * w(j)
        w2 = 0
        # l2 = sum of all (1-P(ij)) * l(j)
        l2 = 0
        for j in range(len(all_individuals_array)):
            num_wins = new_dataframe.at[i, w_array[j]]
            num_matches = new_dataframe.at[i, m_array[j]]
            # ensure that there are matches between the two
            if math.isnan(num_matches) or num_matches == 0:
                continue
            if math.isnan(num_wins):
                num_wins = 0
            pw = num_wins / num_matches
            pl = 1 - pw
            w_opponent = new_dataframe.at[j, "w"]
            l_opponent = new_dataframe.at[j, "l"]
            w2 += pw * w_opponent
            l2 += pl * l_opponent
        new_dataframe.at[i, "w2"] = float(w2)
        new_dataframe.at[i, "l2"] = float(l2)

    # calculating David's score = w + w2 - l - l2
    for i in range(len(all_individuals_array)):
        w = new_dataframe.at[i, "w"]
        l = new_dataframe.at[i, "l"]
        w2 = new_dataframe.at[i, "w2"]
        l2 = new_dataframe.at[i, "l2"]
        ds = w + w2 - l - l2
        final_ds = round(ds, 3)
        new_dataframe.at[i, "DS"] = float(final_ds)

    # necessary output data
    data_to_keep = ["ID", "DS"]
    new_dataframe = new_dataframe[data_to_keep]
    return new_dataframe


def home(df):
    # Format column names
    df.columns = [str(col).lower().strip().replace(" ", "_") for col in df.columns]
    # Remove unneeded columns
    to_keep = ["winner", "loser"]
    # Keep only the specified columns
    df = df[to_keep]
    df = df.dropna(subset=["winner"]).copy()
    # identifies all unique individuals in the data set
    all_individuals = np.array(pd.unique(df[["winner", "loser"]].values.ravel()))
    # sorts individual IDs
    sorted_values = np.sort(all_individuals)
    all_individuals_array = sorted_values.astype(str)
    if df["winner"].isna().any() is True:
        return None

    # creating a dataframe using function.
    new_dataframe, w_array, m_array = creating_new_df(all_individuals_array)
    final_df = solving_for_ds(
        df, all_individuals_array, new_dataframe, w_array, m_array
    )
    return final_df


def determining_function(file_path, filename, output_directory):
    # keep track of pilot:
    if "pilot_1" in filename:
        pilot = "pilot_1."
    elif "pilot_2" in filename:
        pilot = "pilot_2."
    elif "pilot_3" in filename:
        pilot = "pilot_3."
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension == ".csv" or ".xlsx":
        # Check if it's a certain file
        if "urine" in filename:
            assay = "Urine_marking.DS"
            required_columns = ["winner"]

            # reads all the multiple sheets and combines them to one dataframe
            all_sheets = pd.read_excel(file_path, sheet_name=None, header=1)
            relevant_sheets = {
                sheet_name: data
                for sheet_name, data in all_sheets.items()
                if all(column in data.columns for column in required_columns)
            }
            combined_df = pd.concat(relevant_sheets.values(), ignore_index=True)
            # print(urine_tube(combined_df))
            # return urine_tube(combined_df), assay, pilot
            df = urine_tube(combined_df)

        elif "tube" in filename:
            assay = "Tube_test.DS"
            required_columns = ["winner"]
            all_sheets = pd.read_excel(file_path, sheet_name=None, header=0)
            relevant_sheets = {
                sheet_name: data
                for sheet_name, data in all_sheets.items()
                if all(column in data.columns for column in required_columns)
            }
            combined_df = pd.concat(relevant_sheets.values(), ignore_index=True)
            # return urine_tube(combined_df), assay, pilot
            df = urine_tube(combined_df)

        elif "reward" in filename:
            assay = "Reward_competition.DS"
            # return reward_comp(file_path), assay, pilot
            df = reward_comp(file_path)

        elif "home" in filename:
            assay = "Home_cage.DS"
            all_sheets = pd.read_excel(file_path, sheet_name=None, header=1)
            combined_df = pd.concat(all_sheets.values(), ignore_index=True)
            # return home(combined_df), assay, pilot
            df = home(combined_df)

        return df.to_csv(output_directory + "//" + pilot + assay + ".csv", index=False)
        # return df, pilot, assay


def process_files_in_directory(directory_path, target_substring, output_directory):
    # Iterate through each file in the current directory
    stack = [directory_path]

    # create an array to hold dataframes till we can concatenate them together

    while stack:
        current_dir = stack.pop()
        all_dataframes = []
        for filename in os.listdir(current_dir):
            file_path = os.path.join(current_dir, filename)

            if os.path.isdir(file_path):
                # If the item is a directory, add it to the stack
                stack.append(file_path)

            # If the item is a file and contains the target substring, process it
            elif target_substring in filename:
                # DS = David's Score data, assay = assay, pilot = which pilot study
                determining_function(file_path, filename, output_directory)


# extract information from elo sheets
# files are .csv files
def elo_extractor(file_path, filename, output_directory):
    if "pilot_1" in filename:
        pilot = "pilot_1."
        if "Urine" in file_path:
            assay = "Urine_marking.ELO"
        elif "Tube" in file_path:
            assay = "Tube_test.ELO"
        elif "Reward" in file_path:
            assay = "Reward_competition.ELO"
        elif "Home" in file_path:
            assay = "Home_cage.ELO"
    elif "pilot_2" in filename:
        pilot = "pilot_2."
        if "Urine" in file_path:
            assay = "Urine_marking.ELO"
        elif "Tube" in file_path:
            assay = "Tube_test.ELO"
        elif "Reward" in file_path:
            assay = "Reward_competition.ELO"
        elif "Home" in file_path:
            assay = "Home_cage.ELO"
    elif "pilot_3" in filename:
        pilot = "pilot_3."
        if "Urine" in file_path:
            assay = "Urine_marking.ELO"
        elif "Tube" in file_path:
            assay = "Tube_test.ELO"
        elif "Reward" in file_path:
            assay = "Reward_competition.ELO"
        elif "Home" in file_path:
            assay = "Home_cage.ELO"
    df = pd.read_csv(file_path, header=0)
    df = df.copy()
    df.rename(columns={"final_elo_rating": assay}, inplace=True)
    to_keep = ["strain", assay]
    df = df[to_keep]
    return df.to_csv(output_directory + "//" + pilot + assay + ".csv", index=False)
    # return df, pilot, assay


# process all elo files
def elo_processing(directory_path, target_substring, output_directory):
    stack = [directory_path]

    # create an array to hold dataframes till we can concatenate them together

    while stack:
        current_dir = stack.pop()
        all_dataframes = []
        for filename in os.listdir(current_dir):
            file_path = os.path.join(current_dir, filename)

            if os.path.isdir(file_path):
                # If the item is a directory, add it to the stack
                stack.append(file_path)

            # If the item is a file and contains the target substring, process it
            if target_substring in filename:
                # DS = David's Score data, assay = assay, pilot = which pilot study
                elo_extractor(file_path, filename, output_directory)


def initial_merge_csv(input_directory):
    # Iterate through each file in the current directory
    all_files = os.listdir(input_directory)
    processed_files = set()

    for i in all_files:
        # Filter files based on the search string
        pilot = i.split(".")[0]
        assay = i.split(".")[1]
        search_string = pilot + "." + assay
        matching_files = [
            file
            for file in all_files
            if search_string in file and file not in processed_files
        ]
        # Display the matching files
        df_list = []  # Initialize an empty list to store DataFrames for each iteration

        for file in matching_files:
            file_path = os.path.join(input_directory, file)
            calculation_type = file.split(".")[-2]
            if calculation_type == "DS":
                df1 = pd.read_csv(file_path, header=0)
                df1["assay"] = assay
                df1 = pd.concat([df1.pop("assay"), df1], axis=1)
                df_list.append(df1)
            else:
                df2 = pd.read_csv(file_path, header=0)
                df2 = pd.concat([df2.pop("strain"), df2], axis=1)
                df_list.append(df2)

            # Add the processed file to the set
            processed_files.add(file)
        # Check if df_list is not empty before attempting to concatenate
        if df_list:
            # Combine DataFrames from the current iteration
            paired_df = pd.concat(df_list, axis=1)
            paired_df = pd.concat([paired_df.pop("ID"), paired_df], axis=1)
            paired_df.to_csv(
                input_directory + "//" + search_string + "_Combined" + ".csv",
                index=False,
            )

        # Remove files from the current iteration
        for file_name in matching_files:
            os.remove(os.path.join(input_directory, file_name))


# merge all smaller separate files into files for each pilot with all assays in each.
def secondary_merging(input_directory):
    all_files = os.listdir(input_directory)
    processed_files = set()
    for i in all_files:
        # Filter files based on the search string
        pilot = i.split(".")[0]
        matching_files = [
            file for file in all_files if pilot in file and file not in processed_files
        ]
        dataframes = []
        n = 1
        for file in matching_files:
            file_path = os.path.join(input_directory, file)
            df = pd.read_csv(file_path, header=0)
            new_ds = df.at[1, "assay"] + "_DS"
            new_assay = df.at[1, "assay"]
            new_elo = "final_elo_rating" + str(n)
            new_strain = "strain" + str(n)
            df.rename(columns={"DS": new_ds}, inplace=True)
            df.rename(columns={"assay": new_assay}, inplace=True)
            df.rename(columns={"final_elo_rating": new_elo}, inplace=True)
            df.rename(columns={"strain": new_strain}, inplace=True)
            dataframes.append(df)
            processed_files.add(file)
            n += 1
        if len(dataframes) == 0:
            continue
        df = dataframes[0]
        if dataframes:
            merged_df = pd.DataFrame(df)
            for df in dataframes[1:]:
                # Check if 'ID' column exists in both DataFrames before merging
                if "ID" in merged_df.columns and "ID" in df.columns:
                    # Merge DataFrames on 'ID' column using outer join
                    merged_df = pd.merge(merged_df, df, on="ID", how="outer")
                else:
                    print("'ID' column not found in one of the DataFrames.")
                    break
        final_df = merged_df.sort_values(by="ID")

        final_df.to_csv(input_directory + "//" + pilot + ".csv", index=False)
        for file_name in matching_files:
            os.remove(os.path.join(input_directory, file_name))


# combine the pilots into a master file and find correlation matrix
def combine_pilots_correlation(input_directory):
    all_files = os.listdir(input_directory)
    final_df = pd.DataFrame()
    # read all files and concatenate them.
    for i in all_files:
        pilot_name = i.split(".")[0]
        file_path = os.path.join(input_directory, i)
        df = pd.read_csv(file_path, header=0)
        df["pilot"] = pilot_name
        df = pd.concat([df["pilot"], df.drop(columns=["pilot"])], axis=1)
        final_df = pd.concat([final_df, df], ignore_index=True)
        # delete previous files that have their data now combined
        os.remove(os.path.join(input_directory, i))
    strain_columns = final_df.filter(like="strain")
    max_column = strain_columns.apply(lambda col: col.count(), axis=0).idxmax()
    # delete unneeded columns and reorganize columns into correct order
    columns_to_delete = [col for col in strain_columns if col != max_column]
    final_df = final_df.drop(columns=columns_to_delete)
    final_df.rename(columns={max_column: "strain"}, inplace=True)
    final_df = pd.concat([final_df.pop("strain"), final_df], axis=1)
    final_df = pd.concat([final_df.pop("ID"), final_df], axis=1)
    final_df = pd.concat([final_df.pop("pilot"), final_df], axis=1)
    final_df.to_csv(input_directory + "//" + "Master_file" + ".csv", index=False)


# enter the directory path into this variable
# my directory_path = C:\Users\yezon\OneDrive\桌面\Padilla-Coreano_Lab\reformated_data
# my output_path = C:\Users\yezon\OneDrive\Documents\testing_output
directory_path = input("Directory path: ")
output_directory = input("Output Directory: ")
process_files_in_directory(directory_path, "iwata_pilot", output_directory)
elo_processing(directory_path, "final-elo-rating", output_directory)
initial_merge_csv(output_directory)
secondary_merging(output_directory)
combine_pilots_correlation(output_directory)
