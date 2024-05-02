from main import main
import pandas as pd

# loading csv or datasets
# remember to change the path
train_df = pd.read_csv(r"C:\Users\limwe\source\Data\20240303_Korea_Adv_PD.csv", index_col=["PID", "AID"])
test_df = pd.read_csv(r"C:\Users\limwe\source\Data\20240303_Korea_Early_PD.csv", index_col=["PID", "AID"])

# training parameter in Training.py
# change the correct output folder
train_params_dt = {
    "verbose":1,
    "update_params":True,
    "update_file_pth":None,
    "hyper_parameters_json_file_pth":None,
    "grid_search":False,
    "feature_selection":True, 
    "features":None,
    "cv":3,
    "k_features":"best",
    "models":"all",
    "split_test":False,
    "save_trained_model":True,
    "loo": False, 
    "output_folder_pth":r"C:\Users\limwe\source\Output\\",
    "output_file_name":"test_run"
}

# prepared data skip is all is done in the loading csv session
func = lambda x: 0 if x == 0 else 1

train_x = train_df.iloc[:, :-1].values
train_y = train_df.iloc[:, -1].apply(func).values
test_x = test_df.iloc[:, :-1].values
test_y = test_df.iloc[:, -1].apply(func).values

col = train_df.iloc[:, :-1].columns.tolist()

# call main function and hope it work!
main(train_x, train_y, train_params_dt, test_x=test_x, test_y=test_y, columns_name=col)
