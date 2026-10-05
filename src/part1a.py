import os
import sys
import re
import pandas as pd

LABEL_MAP = {
    "AllTapsClosed": 0,
    "Tap1Open": 1,
    "Tap2Open": 2,
    "Tap3Open": 3,
    "Tap4Open": 4,
}

INTERVAL = 0.1

#NOTE: In order for the script to be able to find the data files, the "data" folder needs to be in the same location as the script

# This function is responsible for extracting the sensor number from the root folder (e.g. Sensor1 would return 1)
def get_sensor_number(root_dir):
    dirname = os.path.basename(os.path.normpath(root_dir)) #full path
    match = re.search(r"Sensor(\d+)", dirname, re.IGNORECASE) #search for the string "Sensor" followed by digits
    if not match:
        raise ValueError(f"Cannot find sensor number in directory name: {dirname}")
    return match.group(1)

# This function decides the label of a csv file
def get_label(filename, subdir_name):
    for name, label_id in LABEL_MAP.items(): #loop through every pair in a label map and check if the name is in the filename and if it is return the label id
        if name in filename:
            return label_id
    for name, label_id in LABEL_MAP.items(): #if nothing was found in the filename the second loop checks the subdirectory name for the label
        if name in subdir_name:
            return label_id
    raise ValueError(f"No label found in: {filename}")

# This is the central function of the script. It reads all the csv files, extracts the relevant data, and combines them into one csv file with increasing timestamps. The output file is named ALL_DATA_{sensor_num}.csv and is saved in the same directory as the input folder.
def build_all_data(root_dir):
    sensor_num = get_sensor_number(root_dir) #variable that holds the sensor num
    frames = [] #here we will store the dataframes per csv

    for subdir_name in sorted(os.listdir(root_dir)): #loop throught the subdirectories in the root directory
        subdir_path = os.path.join(root_dir, subdir_name)
        if not os.path.isdir(subdir_path): #skip if it is not a directory
            continue

        for filename in sorted(f for f in os.listdir(subdir_path) if f.endswith(".csv")): #loop through the csv files in sorted order
            try:
                label = get_label(filename, subdir_name) #try to get get the label for the csv
            except ValueError as e:
                print(f"Skipping {filename}: {e}")
                continue

            df = pd.read_csv(os.path.join(subdir_path, filename)) #pandas read the csv file
            df = df[["Time", "Modulus"]].rename(columns={"Time": "ts"}) #keep the columns Time and Modulus and rename Time to ts
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce") #convert ts to numeric, non numeric values will be set to NaN
            df["Modulus"] = pd.to_numeric(df["Modulus"], errors="coerce") #convert Modulus to numeric, non numeric values will be set to NaN
            df = df.dropna().reset_index(drop=True) #drop rows with NaN and reset index
            df["Label"] = label #add Label column
            frames.append(df) #add df to frames

    if not frames: #check whether any valid CSV files are found
        raise ValueError("No CSV files found.")

    # Each file's timestamps start near 0, so shift them to form one increasing timeline across all files.
    offset_frames = [] #here we will save the timestamp shifted versions of the files
    global_end = 0.0 #this variable will hold where the previous file ended in the timeline
    for df in frames: #loop through each cleaned data frame
        df = df.copy() #in order to avoid modifying the original data frame we create a copy
        df["ts"] = df["ts"] - df["ts"].iloc[0] + global_end #shift the timestamps so that they start from where the previous file ended
        global_end = float(df["ts"].iloc[-1]) + INTERVAL #update global_end to be the end of the current file plus the interval so there is no overlap
        offset_frames.append(df)

    combined = pd.concat(offset_frames, ignore_index=True) #we create a large data frame by concatenating all the shifted data frames
    output_path = os.path.join( #building the filename
        os.path.dirname(os.path.normpath(root_dir)),
        f"ALL_DATA_{sensor_num}.CSV",
    )
    combined.to_csv(output_path, index=False) #the combined data frame is saved to a csv
    print(f"Wrote {output_path} ({len(combined)} rows)")
    return output_path, sensor_num #output path of ALL_DATA and the sensor number


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python part1a.py <Sensor_directory>")
        sys.exit(1)

    root_dir = sys.argv[1]
    if not os.path.isdir(root_dir):
        print(f"Directory not found: {root_dir}")
        sys.exit(1)

    build_all_data(root_dir)
