import os
import sys
import numpy as np
import pandas as pd
import re

INTERVAL = 0.1

#NOTE: In order for the script to be able to find the data files, the ALL_DATA_{sensor_num}.CSV file (produced by part1a.py) needs to be accessible at the path provided as an argument

# This function reads ALL_DATA_{sensor_num}.csv, aggregates the data into 0.1 second intervals, and saves the result as AGGR_DATA_{sensor_num}.csv and group rows into 0.1 second windows and save the aggregated values
def aggregate_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.sort_values("ts").reset_index(drop=True)
    # convert to numpy arrays for faster processing
    times = df["ts"].to_numpy(dtype=float)
    values = df["Modulus"].to_numpy(dtype=float)
    labels = df["Label"].to_numpy(dtype=int)

    rows = []
    i = 0
    while i < len(df): #loop until all rows are processed
        start_ts = times[i]
        end_ts = start_ts + INTERVAL #the end of the window is 0.1s after start
        label = labels[i]

        j = i
        while j < len(df) and times[j] <= end_ts and labels[j] == label: #loop until we reach the end of the window or a different label
            j += 1

        window = values[i:j] #current aggregation window
        rows.append((start_ts, np.mean(window), np.min(window), np.max(window), label)) #aggregated row that contains start timestamp, average, min, max, and label
        i = j

    out = pd.DataFrame(rows, columns=["ts", "Average", "Min", "Max", "Label"]) #create a data frame from the aggregated rows
    out.to_csv(output_path, index=False) #the df is written into a csv file
    print(f"Wrote {output_path} ({len(out)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python part1b.py <ALL_DATA_X.CSV>")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    # Get sensor number from filename (e.g. ALL_DATA_1.CSV -> 1)
    basename = os.path.basename(input_path)
    match = re.search(r"ALL_DATA_(\d+)", basename, re.IGNORECASE)
    if not match:
        print(f"Cannot determine sensor number from filename: {basename}")
        sys.exit(1)
    sensor_num = match.group(1)

    output_path = os.path.join(
        os.path.dirname(input_path),
        f"AGGR_DATA_{sensor_num}.csv",
    )
    aggregate_data(input_path, output_path)
