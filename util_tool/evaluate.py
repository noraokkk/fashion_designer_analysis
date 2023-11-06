
import numpy as np
from collections import OrderedDict
import torch
from sklearn import metrics
import torch.nn.functional as F 
from .metrics import custom_mean_avg_precision,subset_accuracy,hamming_loss,example_f1_score
from utils import custom_replace

def compute_metrics(args,all_predictions,all_targets,all_masks,loss,loss_unk,elapsed,known_labels=0,all_metrics=False,verbose=False):
    
    all_predictions = F.sigmoid(all_predictions)
    num_classes = all_predictions.size(1)

    concept_acc = 0
    class_acc = 0

    unknown_label_mask = custom_replace(all_masks,1,0,0)


    if known_labels > 0:
        meanAP = custom_mean_avg_precision(all_targets,all_predictions,unknown_label_mask)
    else:
        meanAP = metrics.average_precision_score(all_targets,all_predictions, average='macro', pos_label=1)
        # meanAP = metrics.average_precision_score(all_targets,all_predictions, average='weighted', pos_label=1)

    optimal_threshold = 0.5 

    all_targets = all_targets.numpy()
    all_predictions = all_predictions.numpy()

    # top_3rd = np.sort(all_predictions)[:,-3].reshape(-1,1)
    # all_predictions_top3 = all_predictions.copy()
    # all_predictions_top3[all_predictions_top3<top_3rd] = 0
    # all_predictions_top3[all_predictions_top3<optimal_threshold] = 0
    # all_predictions_top3[all_predictions_top3>=optimal_threshold] = 1
    #
    # CP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='macro')
    # CR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='macro')
    # CF1_top3 = (2*CP_top3*CR_top3)/(CP_top3+CR_top3)
    # OP_top3 = metrics.precision_score(all_targets, all_predictions_top3, average='micro')
    # OR_top3 = metrics.recall_score(all_targets, all_predictions_top3, average='micro')
    # OF1_top3 = (2*OP_top3*OR_top3)/(OP_top3+OR_top3)

    
    all_predictions_thresh = all_predictions.copy()
    all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
    all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1
    CP = metrics.precision_score(all_targets, all_predictions_thresh, average='macro')
    CR = metrics.recall_score(all_targets, all_predictions_thresh, average='macro')
    CF1 = (2*CP*CR)/(CP+CR)
    OP = metrics.precision_score(all_targets, all_predictions_thresh, average='micro')
    OR = metrics.recall_score(all_targets, all_predictions_thresh, average='micro')
    OF1 = (2*OP*OR)/(OP+OR)  

    acc_ = list(subset_accuracy(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions_thresh, axis=1, per_sample=True))        
    acc = np.mean(acc_)
    hl = np.mean(hl_)
    exf1 = np.mean(exf1_)

    for class_ in range(0, num_classes):
        CPm = metrics.precision_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='macro')
        CRm = metrics.recall_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='macro')
        CF1m = (2 * CPm * CRm) / (CPm + CRm)
        OPm = metrics.precision_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='micro')
        ORm = metrics.recall_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='micro')
        OF1m = (2 * OPm * ORm) / (OPm + ORm)
        WPm = metrics.precision_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='weighted')
        WRm = metrics.recall_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='weighted')
        WF1m = (2 * WPm * WRm) / (WPm + WRm)
        # print(
        #     "class {:d}: CP: {:0.4f} CR: {:0.4f} CF1: {:0.4f} OP: {:0.4f} OR: {:0.4f} OF1: {:0.4f},WP: {:0.4f} WR: {:0.4f} WF1: {:0.4f}".format(
        #         class_, CPm, CRm, CF1m, OPm, ORm, OF1m, WPm, WRm, WF1m))

    eval_ret = OrderedDict([('Subset accuracy', acc),
                        ('Hamming accuracy', 1 - hl),
                        ('Example-based F1', exf1),
                        ('Label-based Micro F1', OF1),
                        ('Label-based Macro F1', CF1)])

    
    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    OF1 = eval_ret['Label-based Micro F1']
    CF1 = eval_ret['Label-based Macro F1']

    if verbose:
        print('loss:  {:0.3f}'.format(loss))
        print('lossu: {:0.3f}'.format(loss_unk))
        print('----')
        print('mAP:   {:0.2f}'.format(meanAP*100))
        print('----')
        print('CP:    {:0.2f}'.format(CP*100))
        print('CR:    {:0.2f}'.format(CR*100))
        print('CF1:   {:0.2f}'.format(CF1*100))
        print('OP:    {:0.2f}'.format(OP*100))
        print('OR:    {:0.2f}'.format(OR*100))
        print('OF1:   {:0.2f}'.format(OF1*100))
        print('WP:    {:0.2f}'.format(WPm*100))
        print('WR:    {:0.2f}'.format(WRm*100))
        print('WF1:   {:0.2f}'.format(WF1m*100))
        # if args.dataset in ['coco','vg']:
        print('----')
        # print('CP_t3: {:0.2f}'.format(CP_top3*100))
        # print('CR_t3: {:0.2f}'.format(CR_top3*100))
        # print('CF1_t3:{:0.2f}'.format(CF1_top3*100))
        # print('OP_t3: {:0.2f}'.format(OP_top3*100))
        # print('OR_t3: {:0.2f}'.format(OR_top3*100))
        # print('OF1_t3:{:0.2f}'.format(OF1_top3*100))

    metrics_dict = {}
    metrics_dict['mAP'] = meanAP
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['OF1'] = OF1
    metrics_dict['CF1'] = CF1
    metrics_dict['loss'] = loss
    metrics_dict['time'] = elapsed

    return metrics_dict


def compute_metrics_normal(all_predictions, all_targets, verbose=False):
    # all_predictions = F.sigmoid(all_predictions)
    num_classes = all_predictions.size(1)
    meanAP = metrics.average_precision_score(all_targets, all_predictions, average='macro', pos_label=1)
    optimal_threshold = 0.5
    all_targets = all_targets.numpy()
    all_predictions = all_predictions.numpy()
    all_predictions_thresh = all_predictions.copy()
    all_predictions_thresh[all_predictions_thresh < optimal_threshold] = 0
    all_predictions_thresh[all_predictions_thresh >= optimal_threshold] = 1
    CP = metrics.precision_score(all_targets, all_predictions_thresh, average='macro')
    CR = metrics.recall_score(all_targets, all_predictions_thresh, average='macro')
    CF1 = (2 * CP * CR) / (CP + CR)
    OP = metrics.precision_score(all_targets, all_predictions_thresh, average='micro')
    OR = metrics.recall_score(all_targets, all_predictions_thresh, average='micro')
    OF1 = (2 * OP * OR) / (OP + OR)

    acc_ = list(subset_accuracy(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    hl_ = list(hamming_loss(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    exf1_ = list(example_f1_score(all_targets, all_predictions_thresh, axis=1, per_sample=True))
    acc = np.mean(acc_)
    hl = np.mean(hl_)
    exf1 = np.mean(exf1_)

    for class_ in range(0, num_classes):
        CPm = metrics.precision_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='macro')
        CRm = metrics.recall_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='macro')
        CF1m = (2 * CPm * CRm) / (CPm + CRm)
        OPm = metrics.precision_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='micro')
        ORm = metrics.recall_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='micro')
        OF1m = (2 * OPm * ORm) / (OPm + ORm)
        WPm = metrics.precision_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='weighted')
        WRm = metrics.recall_score(all_targets[:, class_], all_predictions_thresh[:, class_], average='weighted')
        WF1m = (2 * WPm * WRm) / (WPm + WRm)
        # print(
        #     "class {:d}: CP: {:0.4f} CR: {:0.4f} CF1: {:0.4f} OP: {:0.4f} OR: {:0.4f} OF1: {:0.4f},WP: {:0.4f} WR: {:0.4f} WF1: {:0.4f}".format(
        #         class_, CPm, CRm, CF1m, OPm, ORm, OF1m, WPm, WRm, WF1m))

    eval_ret = OrderedDict([('Subset accuracy', acc),
                            ('Hamming accuracy', 1 - hl),
                            ('Example-based F1', exf1),
                            ('Label-based Micro F1', OF1),
                            ('Label-based Macro F1', CF1)])

    ACC = eval_ret['Subset accuracy']
    HA = eval_ret['Hamming accuracy']
    ebF1 = eval_ret['Example-based F1']
    OF1 = eval_ret['Label-based Micro F1']
    CF1 = eval_ret['Label-based Macro F1']

    if verbose:
        print('----')
        print('mAP:   {:0.2f}'.format(meanAP * 100))
        print('ACC:   {:0.2f}'.format(ACC * 100))
        print('----')
        # print('CP:    {:0.2f}'.format(CP * 100))
        # print('CR:    {:0.2f}'.format(CR * 100))
        # print('CF1:   {:0.2f}'.format(CF1 * 100))
        # print('OP:    {:0.2f}'.format(OP * 100))
        # print('OR:    {:0.2f}'.format(OR * 100))
        # print('OF1:   {:0.2f}'.format(OF1 * 100))
        # print('WP:    {:0.2f}'.format(WPm * 100))
        # print('WR:    {:0.2f}'.format(WRm * 100))
        # print('WF1:   {:0.2f}'.format(WF1m * 100))
        # if args.dataset in ['coco','vg']:
        print('----')
        # print('CP_t3: {:0.2f}'.format(CP_top3*100))
        # print('CR_t3: {:0.2f}'.format(CR_top3*100))
        # print('CF1_t3:{:0.2f}'.format(CF1_top3*100))
        # print('OP_t3: {:0.2f}'.format(OP_top3*100))
        # print('OR_t3: {:0.2f}'.format(OR_top3*100))
        # print('OF1_t3:{:0.2f}'.format(OF1_top3*100))

    metrics_dict = {}
    metrics_dict['mAP'] = meanAP
    metrics_dict['ACC'] = ACC
    metrics_dict['HA'] = HA
    metrics_dict['ebF1'] = ebF1
    metrics_dict['OF1'] = OF1
    metrics_dict['CF1'] = CF1

    return metrics_dict