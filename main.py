from AUTOAI import run_train, plot_ROC_cruve, plot_confusion_matrix, plot_feature_selection, all_clf_evaluate
import os
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np


def main(train_x, train_y, train_params_dt, test_x=None, test_y=None, columns_name=None):

    # Training 
    output = run_train(train_x, train_y, x_test=test_x, y_test=test_y,
                        **train_params_dt)
    result_pth = output[-1]["result_pth"]
    sfs_pth = output[-1]["sfs_pth"]
    save_pth = output[-1]["save_pth"]
    output_pth = output[-1]["output_pth"]
    col = columns_name

    joblib.dump(output, "{}output.joblib".format(output_pth))
    
    # Evaluation

    _, _, out_dt, gt_dt, _, _ = output
    save_fig_pth = "{}".format(result_pth)
    fpr_dt, tpr_dt, auroc_dt, predict_dt, gt_d, result_dt = all_clf_evaluate(out_dt, gt_dt, evaluate_dataset="valid")

    # Visualize results
    plot_ROC_cruve(fpr_dt, tpr_dt, auroc_dt, title="ROC Curve", n_classes=2, savepth="{}_roc_valid.png".format(save_fig_pth))
    out_df = pd.DataFrame.from_dict(result_dt).T
    out_df.columns = ["AUROC", "F1", "Accuracy", "Precision", "Recall"]
    bar = out_df.plot.bar(figsize=(12,5), )
    plt.ylim([0,1])
    plt.ylabel("Score, %")
    plt.savefig("{}valid_bar.png".format(save_fig_pth), bbox_inches = "tight")
    plt.close()
    out_df.to_csv("{}results.csv".format(result_pth))

    fig, ax = plt.subplots(3,3, figsize=(10, 10))
    for a, ((clf_name, pred), (_, gt)) in enumerate(zip(predict_dt.items(), gt_d.items())):
        plot_confusion_matrix(gt, pred, classes=["Normal", "PD"], ax=ax[int(a/3), a%3], title=clf_name)

    fig.tight_layout()
    plt.savefig("{}valid_cm.png".format(save_fig_pth), bbox_inches = "tight")
    plt.close()
    sfs_dt = {}

    for sfs_file in os.listdir(sfs_pth):
        sfs_result = joblib.load(sfs_pth + sfs_file)
        sfs_dt[sfs_file[:-7]] = sfs_result.k_feature_idx_

    all_clf_sfs_ls = []
    for k, v in sfs_dt.items():
        all_clf_sfs_ls.append(v)

    top_10_features = pd.Series([col[c] for c in np.concatenate(all_clf_sfs_ls)]).value_counts()
    top_10_features.plot.bar(color="r")
    plt.ylabel("Counts")
    plt.title("Task 1 features occurrences after SFS")
    plt.savefig("{}sfs_top10.png".format(save_fig_pth), bbox_inches = "tight")
    plt.close()
