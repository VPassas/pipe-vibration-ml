import sys
import numpy as np
import pandas as pd

INTERVAL = 0.1

#NOTE: In order for the script to be able to find the data files, the "data" folder needs to be in the same location as the script

# This function reads AGGR_DATA_1.csv and AGGR_DATA_2.csv and merges them based on overlapping timestamps and labels and saves the result as AGGR_DATA_ALL.csv. The new file has columns for the average, min, max, and the label of both sensors.
def merge_aggr_data(path1, path2, output_path):
    df1 = pd.read_csv(path1).sort_values("ts").reset_index(drop=True) #after the first aggregated data is read it is then sorted by timestamp and the index is reset
    df2 = pd.read_csv(path2).sort_values("ts").reset_index(drop=True) #the same is done for the second aggregated data
    # we convert the timestamps to numpy arrays, compute the ending time and create a boolean array to track which rows of df2 are used
    times2 = df2["ts"].to_numpy(dtype=float)
    ends2 = times2 + INTERVAL
    used2 = np.zeros(len(df2), dtype=bool)
    rows = []

    for i, r1 in df1.iterrows(): #loop over each row(r1) of the first df and keep the index(_)
        ts1 = float(r1["ts"])
        end1 = ts1 + INTERVAL
        label1 = int(r1["Label"])
        # We create a boolean mask that checks for rows in df2 where the timestamp overlaps with the current row of df1 and the label is the same. To do this we check if the start of the df2 row is before the end of the df1 row and if the end of the df2 row is after the start of the df1 row and the labels match.
        mask = (times2 < end1) & (ends2 > ts1) & (df2["Label"].to_numpy(dtype=int) == label1)
        # If the mask does not have any true values then we append a row to the output with the df1 values and NaNs for the df2 values. If there are true values in the mask then we mark those rows in df2 as used, compute the average, min, and max for those rows, and append a row to the output with both the df1 and aggregated df2 values.
        if not mask.any(): 
            rows.append((ts1,
                         float(r1["Average"]), float(r1["Min"]), float(r1["Max"]),
                         np.nan, np.nan, np.nan,
                         label1))
        else:
            used2 |= mask #if mask is true then used2 is set to true for those rows
            s2 = df2.loc[mask]
            rows.append((ts1,
                         float(r1["Average"]), float(r1["Min"]), float(r1["Max"]),
                         float(s2["Average"].mean()), float(s2["Min"].min()), float(s2["Max"].max()),
                         label1))

    for i, r2 in df2.loc[~used2].iterrows(): #Loop through the rows of df2 that were not used in the previous loop and append them to the output with NaNs for the df1 values
        rows.append((float(r2["ts"]),
                     np.nan, np.nan, np.nan,
                     float(r2["Average"]), float(r2["Min"]), float(r2["Max"]),
                     int(r2["Label"])))
    # A data frame is created from the output rows with columns for the timestamp, average, min, max of both sensors and  label. The data frame is then sorted by timestamp and saved to a csv file.
    out = pd.DataFrame(rows, columns=[
        "ts",
        "Sensor1_Avg", "Sensor1_Min", "Sensor1_Max",
        "Sensor2_Avg", "Sensor2_Min", "Sensor2_Max",
        "Label",
    ])
    out = out.sort_values("ts").reset_index(drop=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} rows)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python part1c.py <AGGR_DATA_1.csv> <AGGR_DATA_2.csv> where the output file will be saved: <AGGR_DATA_ALL.csv>")
        sys.exit(1)

    merge_aggr_data(sys.argv[1], sys.argv[2], sys.argv[3])



