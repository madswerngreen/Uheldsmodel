from Utils import Data, QQ_PLOT_ALL
import pandas as pd
import os

# set wd to script location 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("Current working directory:")
print(os.getcwd())

print("Generating QQ plots for baseline and joint model...")

df_pred = pd.read_csv("../Results/Baseline/all_predictions.csv")
df_params = pd.read_csv("../Results/Baseline/fold_params.csv")
QQ_PLOT_ALL(df_params, df_pred, path = "../Results/Baseline/Figures")

df_pred = pd.read_csv("../Results/JointModel/all_predictions.csv")
df_params = pd.read_csv("../Results/JointModel/fold_params.csv")
QQ_PLOT_ALL(df_params, df_pred, path = "../Results/JointModel/Figures")