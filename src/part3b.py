import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


WINDOW = 100  # 100 rows x 0.1s = 10-second rolling window
Path = r"E:\dereeHomeworksosto\compsciKaiMeta\ITC4568ML\project\src\data\AGGR_DATA_ALL.csv"  # Update this path to the specific AGGR_DATA_ALL.csv path

# This function removes unnecessary columns, splits the set into training, validation, and test, fills missing values and scales features
def split_and_scale(df):
    df = df.drop(columns=["ts"]).reset_index(drop=True) #drop ts

    X = df.drop(columns=["Label"]).values #feature matrix created by dropping the label column and converting to numpy array
    y = df["Label"].values.astype(int) #target vector is the label coumn converted to numpy array of integers
    #The data is split into training, validation and test sets using stratified sampling to keep the class distribution between sets. Random state makes the split repeatable
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


# Part 3B - alternative topologies and 1D-CNN
print("Part 3B - Alternative Topologies and 1D-CNN")

df_raw = pd.read_csv(Path) #read the aggregated data
df_eng = engineer_features(df_raw) #df_eng holds the data frame with the engineered features
d3 = split_and_scale(df_eng) #d3 contains the training, validation and test sets with both the original and scaled features but this time with the engineered features included


print("Topology A - MLP (64, 32), tanh")
#Evaluating an MLP with two hidden layers of 64 and 32 nodes, tanh activation function, Adam optimizer, max 1000 iterations, random state for reproducibility, early stopping with a validation fraction of 0.15
p3b_a = evaluate_mlp(
    MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="tanh", solver="adam",
        max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.15,
    ),
    d3["X_train_s"], d3["y_train"],
    d3["X_val_s"],   d3["y_val"],
    d3["X_test_s"],  d3["y_test"],
    name="Topology A: MLP (64, 32), tanh",
)

print("Topology B - MLP (128), sigmoid")
#Evaluating an MLP with one hidden layer of 128 nodes, sigmoid activation function, Adam optimizer, max 1000 iterations, random state for reproducibility, early stopping with a validation fraction of 0.15
p3b_b = evaluate_mlp(
    MLPClassifier(
        hidden_layer_sizes=(128,), activation="logistic", solver="adam",
        max_iter=1000, random_state=42, early_stopping=True, validation_fraction=0.15,
    ),
    d3["X_train_s"], d3["y_train"],
    d3["X_val_s"],   d3["y_val"],
    d3["X_test_s"],  d3["y_test"],
    name="Topology B: MLP (128), sigmoid",
)

print("1D-CNN (PyTorch)")

X_train_t = torch.FloatTensor(d3["X_train_s"]).unsqueeze(1) #convert scaled training features to a PyTorch float tensor. unsqueeze(1) adds a channel dimension so that CNN input shape fits Conv1D
X_val_t   = torch.FloatTensor(d3["X_val_s"]).unsqueeze(1) #convert scaled validation features to a PyTorch float tensor
X_test_t  = torch.FloatTensor(d3["X_test_s"]).unsqueeze(1) #convert scaled test features to a PyTorch float tensor
y_train_t = torch.LongTensor(d3["y_train"]) #convert training labels to a PyTorch long tensor
y_val_t   = torch.LongTensor(d3["y_val"]) #convert validation labels to a PyTorch long tensor
y_test_t  = torch.LongTensor(d3["y_test"]) #convert test labels to a PyTorch long tensor
# A training data loader is created by feeding training data in batches of 64 rows and shuffling batches each epoch for memory efficiency
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
num_features = d3["X_train_s"].shape[1] #number of input features

# This class defines a 1D Convolutional Neural Network for classification.
class Conv1DClassifier(nn.Module):
    def __init__(self, num_features, num_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), #1 input channel because we unsqueezed 1, padding of 1 to keep output same length as input
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes), #num_classes is 5 because we have 5 labels (0 to 4)
        )

    def forward(self, x):
        return self.fc(self.conv(x)) #forward pass through the convolutional layers and then the fully connected layers


model     = Conv1DClassifier(num_features=num_features, num_classes=5) #creating an instance of the CNN
criterion = nn.CrossEntropyLoss() #loss function
optimizer = optim.Adam(model.parameters(), lr=0.001) #optimizer is Adam with a learning rate of 0.001 it will update the model parameters based on the computed gradients during training

best_val_loss   = float("inf") #stores bust validation loss thus far
patience        = 15 #epochs for early stopping
patience_counter = 0 #number of bad epochs in a row
best_state      = None #best model weights

for epoch in range(200): #loop for a maximum of 200 epochs most likely early stopping won't let it reach that point
    model.train() #training mode
    train_loss = 0.0 #total training loss for epocj
    for X_batch, y_batch in train_loader: #loop through batches of training data
        optimizer.zero_grad() #reset gradients
        loss = criterion(model(X_batch), y_batch) #feed batch through the model and compute loss
        loss.backward() #use backprop to compute gradients
        optimizer.step() #update model weights
        train_loss += loss.item() * len(y_batch) #adds this batch's loss to the total
    train_loss /= len(train_loader.dataset) #divides train_loss by the total number of training samples to get the average loss for the epoch

    model.eval() #evaluation mode
    with torch.no_grad(): #turn off gradient tracking for memory efficiency
        val_out  = model(X_val_t) #feed the validation set to the model
        val_loss = criterion(val_out, y_val_t).item() #compute val loss
        val_acc_e = accuracy_score(d3["y_val"], val_out.argmax(dim=1).numpy()) #compute validation accuracy by gettin the predicted class for each validation row

    if val_loss < best_val_loss: #did this epoch improve the validation loss?
        best_val_loss    = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()} #save copy of the model weights of the best version
    else: #validation loss did not improve, increase patience counter
        patience_counter += 1

    if (epoch + 1) % 20 == 0: #print progress for every 20 epochs
        print(f"Epoch {epoch+1:3d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc_e:.4f}")

    if patience_counter >= patience: #checks if the model has failed to improve for too many epochs
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state) #load the best model weights

model.eval() #evaluation mode
with torch.no_grad():
    val_preds  = model(X_val_t).argmax(dim=1).numpy() #predict validation classes
    cnn_val_acc = accuracy_score(d3["y_val"], val_preds) #compute validation accuracy
    cnn_val_f1  = f1_score(d3["y_val"], val_preds, average="weighted") #compute validation F1
    print(f"\nCNN Validation - Accuracy: {cnn_val_acc:.4f}, F1: {cnn_val_f1:.4f}")
    print(classification_report(d3["y_val"], val_preds, zero_division=0))

    test_preds  = model(X_test_t).argmax(dim=1).numpy() #predict test classes
    cnn_test_acc = accuracy_score(d3["y_test"], test_preds) #compute test accuracy
    cnn_test_f1  = f1_score(d3["y_test"], test_preds, average="weighted") #compute test F1
    print(f"CNN Test - Accuracy: {cnn_test_acc:.4f}, F1: {cnn_test_f1:.4f}")
    print(classification_report(d3["y_test"], test_preds, zero_division=0))

p3b_cnn = {"val_acc": cnn_val_acc, "val_f1": cnn_val_f1, "test_acc": cnn_test_acc, "test_f1": cnn_test_f1}
