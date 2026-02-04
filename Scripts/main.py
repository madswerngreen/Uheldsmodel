#Name = main.py

# import modules from utils
from Utils import Data, cross_validate_joint, QQ_PLOT_ALL, Model, build_param_table

import os
# set wd to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def baseline():
    """ Fit baseline model without any features for comparison """
    DATA_PATH = "../Data/Model_Input/Uheld_LINKS.csv"
    
    # -------------------------------------------------------
    # 1. Load data & initialize feature sets
    # -------------------------------------------------------
    print("📂 Loading data...")
    data_obj = Data(path=DATA_PATH, 
                    FEATURES_X = [], 
                    FEATURES_Z = [], 
                    FEATURES_W = [])
    
    # -------------------------------------------------------
    # 2. Cross-validation
    # -------------------------------------------------------
    print("\n📊 Running cross-validation on model...")
    df_cv, df_pred, df_params = cross_validate_joint(
        data_obj,
        K=5,
        out_dir="../Results/Baseline"
    )
    return {
        "cv_metrics": df_cv,
        "cv_predictions": df_pred,
        "cv_params": df_params
    }
    # -------------------------------------------------------
def main():
    """ Main function to fit joint model with features in cross-validation setup"""
    DATA_PATH = "../Data/Model_Input/Uheld_LINKS.csv"

    print("\n==============================")
    print("   JOINT MODEL ")
    print("==============================\n")

    # -------------------------------------------------------
    # 1. Load data & initialize feature sets
    # -------------------------------------------------------
    print("📂 Loading data...")
    data_obj = Data(path=DATA_PATH)

    # -------------------------------------------------------
    # 3. Cross-validation
    # -------------------------------------------------------
    print("\n📊 Running cross-validation on model...")
    df_cv, df_pred, df_params = cross_validate_joint(
        data_obj,
        K=5,
        out_dir="../Results/JointModel"
    )

    print("\n===== FINAL CROSS-VALIDATION RESULTS =====")
    print(df_cv.mean(numeric_only=True).round(3))

    print("\nAll results saved to ../Results/JointModel_Reduced")

    return {
        "cv_metrics": df_cv,
        "cv_predictions": df_pred,
        "cv_params": df_params
    }

def fit_and_save_params():
    """ Fit model on all data and save parameter table """
    DATA_PATH = "../Data/Model_Input/Uheld_LINKS.csv"
    print("\nFitting model on all data...")
    data_obj = Data(path=DATA_PATH)
    y, y_p, z, X, Z, W, exposure = data_obj.var_extract()
    feature_names_X = ["Intercept"] + data_obj.FEATURES_X
    feature_names_Z = ["Intercept"] + data_obj.FEATURES_Z
    feature_names_W = ["Intercept"] + data_obj.FEATURES_W
    model = Model()
    model.optimize(y, y_p, z, X, Z, W, exposure, verbose=True)
    df_params = build_param_table(model, feature_names_X, feature_names_Z, feature_names_W)
    out_path = "../Results/all_data_params.csv"
    df_params.to_csv(out_path, index=False)
    print(f"\nParameter table saved to {out_path}")
    
if __name__ == "__main__":
    results = main()
    QQ_PLOT_ALL(results["cv_params"], results["cv_predictions"],
                    path = "../Results/JointModel/Figures")
    
    results = baseline()
    QQ_PLOT_ALL(results["cv_params"], results["cv_predictions"],
                    path = "../Results/Baseline/Figures")
    
    fit_and_save_params()