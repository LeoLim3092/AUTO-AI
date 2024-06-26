import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, precision_score, recall_score


def single_clf_evaluation(gt, proba, clf=""):

    fpr, tpr, thresholds = roc_curve(gt, proba)
    J_stats = tpr - fpr
    opt_thresholds = thresholds[np.argmax(J_stats)]
    print(clf + 'Optimum threshold ' + str(opt_thresholds))
    predict_label = proba > opt_thresholds

    auroc = roc_auc_score(gt, proba)
    f1 = f1_score(gt, predict_label)
    acc = accuracy_score(gt, predict_label)
    precision = precision_score(gt, predict_label)
    recall = recall_score(gt, predict_label)

    return fpr, tpr, auroc, f1, acc, precision, recall, predict_label


def fold_evaluation(p_ls, gt_ls):
    aucs = []
    aps = []
    accs = []
    f1s = []
    tprs = []
    fprs = []
    P = []
    best_ths = []
    recalls = []
    precisions = []

    for _p, _gt in zip(p_ls, gt_ls):

        if isinstance(_gt, list):
            _gt = np.array(_gt).reshape(-1, 1)

        if len(_gt.shape) == 1:
            _gt = _gt.reshape(-1, 1)

        if _gt.shape[1] > 1:
            y = np.argmax(_gt, axis=1).ravel()
        else:
            y = _gt.ravel()

        if _p.shape[1] == 2:
            _p = _p[:, 1]

        no_classes = len(np.unique(y))
        base_fpr = np.linspace(0, 1, 101)
        false_positive_rate, true_positive_rate, threshold = roc_curve(_gt, _p)
        interp_tpr = np.interp(base_fpr, false_positive_rate, true_positive_rate)
        interp_tpr[0] = 0.0
        roc_area_under_curve = roc_auc_score(_gt, _p)

        aucs.append(roc_area_under_curve)
        fpr = false_positive_rate
        tpr = true_positive_rate

        i = np.arange(len(tpr))
        roc_df = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold,
                                                                                                 index=i)})
        roc_t = roc_df.iloc[(roc_df.tf - 0).abs().argsort()[:1]]
        best_th = roc_t['threshold'].values
        best_th_idx = roc_t.index

        tprs.append(interp_tpr)
        fprs.append(base_fpr)

        _P = [1 if P >= best_th else 0 for P in _p]
        accs.append(accuracy_score(y, _P))
        f1s.append(f1_score(y, _P))
        P.append(_P)
        recalls.append(recall_score(y, _P))
        best_ths.append(best_th_idx)
        precisions.append(precision_score(y, _P))

    return accs, f1s, aucs, aps, fprs, tprs, P, recalls, precisions, best_ths


def all_clf_evaluate(out_dt, gt_dt, evaluate_dataset="valid", test_select="best"):
    result_dt = {}
    fpr_dt = {}
    tpr_dt = {}
    auroc_dt = {}
    predict_dt = {}
    gt_d = {}

    for (dataset, proba_dt), (_, gt_dict) in zip(out_dt.items(), gt_dt.items()):
        if evaluate_dataset == "valid" and dataset == "valid":
            for (clf_name, proba), (_, gt) in zip(proba_dt.items(), gt_dict.items()):
                
                if isinstance(proba, list):
                    fpr, tpr, auroc, f1, acc, precision, recall, predict = single_clf_evaluation(np.concatenate(gt),
                                                                                    np.concatenate(proba)[:, 1], clf=clf_name)
                    gt_d[clf_name] = np.concatenate(gt)
                else:
                    fpr, tpr, auroc, f1, acc, precision, recall, predict = single_clf_evaluation(gt, proba[:, 1], clf=clf_name)
                    gt_d[clf_name] = gt
                    
                fpr_dt[clf_name] = fpr
                tpr_dt[clf_name] = tpr
                auroc_dt[clf_name] = auroc
                result_dt[clf_name] = [auroc, f1, acc, precision, recall]
                predict_dt[clf_name] = predict

        elif evaluate_dataset == "train" and dataset == "train":
            for (clf_name, proba), (_, gt) in zip(proba_dt.items(), gt_dict.items()):
                fpr, tpr, auroc, f1, acc, precision, recall, predict = single_clf_evaluation(np.concatenate(gt),
                                                                                  np.concatenate(proba)[:, 1], clf=clf_name)
                fpr_dt[clf_name] = fpr
                tpr_dt[clf_name] = tpr
                auroc_dt[clf_name] = auroc
                result_dt[clf_name] = [auroc, f1, acc, precision, recall]
                predict_dt[clf_name] = predict
                gt_d[clf_name] = np.concatenate(gt)

        elif evaluate_dataset == "test" and dataset == "test":
            for (clf_name, proba), (_, gt) in zip(proba_dt.items(), gt_dict.items()):
                print(clf_name)
                
                if test_select == "best":
                    best_auroc = 0
                    best_idx = 0
                    for i, (g, p) in enumerate(zip(gt, proba)):
                        _, _, auroc, _, _, _, _, _ = single_clf_evaluation(g, p[:, 1], clf=clf_name)
                        if auroc > best_auroc:
                            best_auroc = auroc
                            best_idx = i
                    print(f'best idx:{best_idx}')

                    fpr, tpr, auroc, f1, acc, precision, recall, predict = single_clf_evaluation(gt[best_idx],
                                                                                      proba[best_idx][:, 1], clf=clf_name)
                    fpr_dt[clf_name] = fpr
                    tpr_dt[clf_name] = tpr
                    auroc_dt[clf_name] = auroc
                    result_dt[clf_name] = [auroc, f1, acc, precision, recall]
                    predict_dt[clf_name] = predict
                    gt_d[clf_name] = gt[best_idx]

                elif test_select == "all_fold":
                    accs, f3s, aucs, aps, fprs, tprs, predict, recalls, precisions, best_ths = fold_evaluation(proba,
                                                                                                               gt, clf=clf_name)
                    fpr_dt[clf_name] = np.mean(fprs, axis=0)
                    tpr_dt[clf_name] = np.mean(tprs, axis=0)
                    auroc_dt[clf_name] = np.mean(aucs, axis=0)
                    result_dt[clf_name] = [np.mean(aucs), np.mean(f3s), np.mean(accs), np.mean(precisions),
                                           np.mean(recalls)]
                    predict_dt[clf_name] = predict
                    gt_d[clf_name] = gt
                    
                elif test_select == "ensemble":
                    for (clf_name, proba), (_, gt) in zip(proba_dt.items(), gt_dict.items()):

                        fpr, tpr, auroc, f1, acc, precision, recall, predict = single_clf_evaluation(gt,
                                                                                          proba[:, 1], clf=clf_name)
                        fpr_dt[clf_name] = fpr
                        tpr_dt[clf_name] = tpr
                        auroc_dt[clf_name] = auroc
                        result_dt[clf_name] = [auroc, f1, acc, precision, recall]
                        predict_dt[clf_name] = predict
                        gt_d[clf_name] = gt
                        
                elif test_select == "mean":

                    fpr, tpr, auroc, f1, acc, precision, recall, predict = single_clf_evaluation(gt[0], np.mean(proba, 0)[:, 1], clf=clf_name)
                    print(np.mean(proba, 0).sum(1))
                    fpr_dt[clf_name] = fpr
                    tpr_dt[clf_name] = tpr
                    auroc_dt[clf_name] = auroc
                    result_dt[clf_name] = [auroc, f1, acc, precision, recall]
                    predict_dt[clf_name] = predict
                    gt_d[clf_name] = gt[0]

    return fpr_dt, tpr_dt, auroc_dt, predict_dt, gt_d, result_dt
