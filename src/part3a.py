import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


WINDOW = 100  # 100 rows x 0.1s = 10-second rolling window
Path = r""  # Update this path to the specific AGGR_DATA_ALL.csv path

# This function removes unnecessary columns, splits the set into training, validation, and test, fills missing values and scales features
def split_and_scale(df):
    df = df.drop(columns=["ts"]).reset_index(drop=True) #drop ts

    X = df.drop(columns=["Label"]).values #feature matrix created by dropping the label column and converting to numpy array
    y = df["Label"].values.astype(int) #target vector is the label coumn converted to numpy array of integers
    #The data is split into training, validation and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.6, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # Impute NaN values that come from the non overlapping sensor intervals with the mean.
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)
    X_test  = imputer.transform(X_test)
    # A scaler is created and will scale each feature so it has roughly mean 0 and std dev 1
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    # Dictionary with all the prepared datasets
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_s": X_train_s, "X_val_s": X_val_s, "X_test_s": X_test_s,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "X_eval": np.concatenate([X_val, X_test]),
        "y_eval": np.concatenate([y_val, y_test]),
    }

# This function trains and evaluates an MLP model
def evaluate_mlp(model, X_tr, y_tr, X_v, y_v, X_te, y_te, name):
    model.fit(X_tr, y_tr) #model is trained on the training set

    val_acc = accuracy_score(y_v, model.predict(X_v)) #validation accuracy is computed by comparing the true labels of the validation set with the predicted labels from the model
    val_f1  = f1_score(y_v, model.predict(X_v), average="weighted") #validation F1 score is computed with the same way but this time the f1_score function with the weighted average
    y_pred  = model.predict(X_te) #Predict labels for the test set
    test_acc = accuracy_score(y_te, y_pred) #test accuracy
    test_f1  = f1_score(y_te, y_pred, average="weighted") #test F1 score with weighted average

    print(f"\n{name}")
    print(f"  Validation - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    print(f"  Test       - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(classification_report(y_te, y_pred, zero_division=0))
    # Return a dictionary with validation and  test metrics
    return {"val_acc": val_acc, "val_f1": val_f1, "test_acc": test_acc, "test_f1": test_f1}

# This function trains and evaluates a Decision Tree model. The model trains on X_tr and y_tr and evaluates on X_eval
def evaluate_dt(model, X_tr, y_tr, X_eval, y_eval, name):
    model.fit(X_tr, y_tr) #decision tree is trained on the training set

    y_pred   = model.predict(X_eval) #predict labels for the eval set
    eval_acc = accuracy_score(y_eval, y_pred) #evaluation accuracy
    eval_f1  = f1_score(y_eval, y_pred, average="weighted") #evaluation F1 score with weighted average

    print(f"\n{name}")
    print(f"  Eval (val+test) - Accuracy: {eval_acc:.4f}, F1: {eval_f1:.4f}")
    print(classification_report(y_eval, y_pred, zero_division=0))
    # Dictionary with evaluation accuracy and F1 score
    return {"eval_acc": eval_acc, "eval_f1": eval_f1}

# This function creates new features from the original sensor features
def engineer_features(df):
    df = df.copy()

    df["S1_Range"]    = df["Sensor1_Max"] - df["Sensor1_Min"] #range of sensor 1
    df["S2_Range"]    = df["Sensor2_Max"] - df["Sensor2_Min"] #range of sensor 2
    df["S1_AbsMax"]   = df[["Sensor1_Min", "Sensor1_Max"]].abs().max(axis=1) #largest absolute value of sensor 1
    df["S2_AbsMax"]   = df[["Sensor2_Min", "Sensor2_Max"]].abs().max(axis=1) #largest absolute value of sensor 2
    df["Range_Diff"]  = df["S1_Range"] - df["S2_Range"] #difference in range between the two sensors
    df["AbsMax_Diff"] = df["S1_AbsMax"] - df["S2_AbsMax"] #difference in absolute max value between the two sensors

    # Group consecutive rows with the same label so rolling windows don't cross boundaries
    df["seg"] = (df["Label"] != df["Label"].shift()).cumsum()

    # Largest absolute value in the previous 10 seconds.
    # This is done by grouping the data frame by "seg" and applying a rolling max with a window of 100 rows (10 seconds) that is shifted by 1 row so that we will not include the current row in the calculation.
    # fillna(0) is used to fill the first 10 rows of each segment where there are not enough previous rows to calculate the rolling max.
    df["S1_AbsMax_Roll10s"] = df.groupby("seg")["S1_AbsMax"].transform(
        lambda x: x.shift(1).rolling(WINDOW, min_periods=1).max()
    ).fillna(0)
    df["S2_AbsMax_Roll10s"] = df.groupby("seg")["S2_AbsMax"].transform(
        lambda x: x.shift(1).rolling(WINDOW, min_periods=1).max()
    ).fillna(0)
    # Loop through the average and max columns of both sensors and create new features for the rolling standard deviation and mean over the previous 10 seconds.
    # This loop will show how much Sensor1_Avg has been fluctuating in the last 10s
    for col in ["Sensor1_Avg", "Sensor1_Max", "Sensor2_Avg", "Sensor2_Max"]:
        df[f"{col}_RollStd"] = df.groupby("seg")[col].transform(
            lambda x: x.shift(1).rolling(WINDOW, min_periods=1).std()
        ).fillna(0)
    # Loop through the min and max columns of both sensors and create new features for the rolling mean over the previous 10 seconds.
    # This loop will show what the typical peak of sensor 1 has been in the last 10s
    for col in ["Sensor1_Min", "Sensor1_Max", "Sensor2_Min", "Sensor2_Max"]:
        df[f"{col}_RollMean"] = df.groupby("seg")[col].transform(
            lambda x: x.shift(1).rolling(WINDOW, min_periods=1).mean()
        ).fillna(0)

    df = df.drop(columns=["seg"]) #remove the temporary segment column
    return df


print("Part 3A - Engineered Features")

df_raw = pd.read_csv(Path) #read the aggregated data
df_eng = engineer_features(df_raw) #df_eng holds the data frame with the engineered features
d3 = split_and_scale(df_eng) #d3 contains the training, validation and test sets with both the original and scaled features but this time with the engineered features included

print("MLP - 10 hidden nodes, ReLU (engineered features)")
p3a_mlp = evaluate_mlp( #same process as before but now with the engineered features
    MLPClassifier(
        hidden_layer_sizes=(10,), activation="relu", solver="adam",
        max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.15,
    ),
    d3["X_train_s"], d3["y_train"],
    d3["X_val_s"],   d3["y_val"],
    d3["X_test_s"],  d3["y_test"],
    name="MLP (10 hidden, ReLU) - engineered",
)

print("Decision Tree (engineered features)")
p3a_dt = evaluate_dt( #same process as before but now with the engineered features
    DecisionTreeClassifier(random_state=42),
    d3["X_train"], d3["y_train"],
    d3["X_eval"],  d3["y_eval"],
    name="Decision Tree - engineered",
)
