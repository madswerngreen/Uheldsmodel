# main.py
from Utils import Data, cross_validate_joint, QQ_PLOT_ALL

def fit_baseline():
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


if __name__ == "__main__":
    results = main()
    QQ_PLOT_ALL(results["cv_params"], results["cv_predictions"],
                    path = "../Results/JointModel/Figures")
    
    results = fit_baseline()
    QQ_PLOT_ALL(results["cv_params"], results["cv_predictions"],
                    path = "../Results/Baseline/Figures")
