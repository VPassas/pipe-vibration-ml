import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


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


print("Part 2 - Training on Raw Data")

df_raw = pd.read_csv(Path) #read the aggregated data
print(f"Loaded {len(df_raw)} rows, {df_raw.shape[1]} columns")

d2 = split_and_scale(df_raw) #d2 contains the training, validation and test sets with both the original and scaled features

print("MLP - 10 hidden nodes, ReLU activation")
p2_mlp = evaluate_mlp( #evaluate MLP model
    MLPClassifier( #create an MLPClassifier with 10 hidden nodes, ReLU activation, Adam optimizer, max 1000 iterations, random state for reproducibility, early stopping with a validation fraction of 0.15
        hidden_layer_sizes=(10,), activation="relu", solver="adam",
        max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.15,
    ),
    #passing scaled training, validation and test sets to the evaluation function
    d2["X_train_s"], d2["y_train"],
    d2["X_val_s"],   d2["y_val"],
    d2["X_test_s"],  d2["y_test"],
    name="MLP (10 hidden, ReLU)",
)

print("Decision Tree")
p2_dt = evaluate_dt( #evaluate Decision Tree model
    DecisionTreeClassifier(random_state=42),
    #raw training and combined evaluation data
    d2["X_train"], d2["y_train"],
    d2["X_eval"],  d2["y_eval"],
    name="Decision Tree",
)
