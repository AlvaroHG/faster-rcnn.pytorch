import numpy as np
from collections import defaultdict
import os
from tqdm import tqdm
from ai2io import *
import copy
import cv2


def convert_to_my_format_from_predicted(predicted_bbs):
    """convert (ctr_x, ctr_y, w, h) -> (x1, y1, x2, y2)"""
    output_bb = []
    for predicted_bb in predicted_bbs:
        x = int(predicted_bb[1])
        y = int(predicted_bb[2])
        w = int(predicted_bb[3] / 2)
        h = int(predicted_bb[4] / 2)
        output_bb.append([predicted_bb[0], x - w, y - h, x + w, y + h, predicted_bb[5]])
    return output_bb

def dets_to_bb(dets):
    output_bbs = []
    for k in range(dets.shape[0]):
        output_bbs.append([dets[k, 0] + 1, dets[k, 1] + 1, dets[k, 2] + 1, dets[k, 3] + 1, dets[k, -1]])
    return output_bbs


def compute_intersect(a, b):
    # format of bb: [ x1, y1, x2, y2 ]
    top = max(a[1], b[1])
    bottom = min(a[3], b[3])
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    tb = bottom - top  # top to bottom
    lr = right - left  # left to right
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection


def compute_iofirst(a, b):
    a_w = a[2] - a[0]
    a_h = a[3] - a[1]
    a_area = a_w * a_h
    if a_area == 0:
        return 0
    intersection = compute_intersect(a, b)
    return intersection / (a_area + 0.)


def _compute_iou(a_bb, b_bb):
    # format of bb: [ x1, y1, x2, y2 ]
    top = max(a_bb[1], b_bb[1])
    bottom = min(a_bb[3], b_bb[3])
    left = max(a_bb[0], b_bb[0])
    right = min(a_bb[2], b_bb[2])
    a_bb_w = a_bb[2] - a_bb[0]
    a_bb_h = a_bb[3] - a_bb[1]
    b_bb_w = b_bb[2] - b_bb[0]
    b_bb_h = b_bb[3] - b_bb[1]
    tb = bottom - top  # top to bottom
    lr = right - left  # left to right
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr

    return intersection / (a_bb_w * a_bb_h + b_bb_w * b_bb_h - intersection + 0.)


def yolo_bb_to_eval_bb(yolo_bb):
    '''
    If yolo_bb is list, return list. If it is a one value, return one value.
    yolo_bb: [start_x, start_y, width, height]
    return: ['x1', 'x2', 'y1', 'y2']
    '''
    eval_format = [yolo_bb[0] - yolo_bb[2] / 2, yolo_bb[1] - yolo_bb[3] / 2,
                   yolo_bb[0] + yolo_bb[2] / 2, yolo_bb[1] + yolo_bb[3] / 2, yolo_bb[4]]
    return eval_format


def convert_yolo_to_eval_format(yolo_format):
    def _pred_to_cat(predicts):
        categories = set()
        predict_cat = defaultdict(list)  # todo: getting from config
        for predict in predicts:
            cat_name = predict[0]
            categories.add(cat_name)
            predict_cat[cat_name].append(predict[1:])
        return predict_cat, categories

    #
    predicts_cat, categories = _pred_to_cat(yolo_format)
    for bb_type in categories:
        for i, bb in enumerate(predicts_cat[bb_type]):
            predicts_cat[bb_type][i] = yolo_bb_to_eval_bb(bb)
    return predicts_cat


def _compute_max_iou(bb_a, bb_b):
    '''
    compute max iou from bb_a (a single sample) to bb_b (list)
    :param gnd_bb: ['x1', 'x2', 'y1', 'y2']
    :param predict_bbs: list of ['x1', 'x2', 'y1', 'y2']
    :return: iou
    '''
    ious = []
    if len(bb_b) == 0:
        return 0, 0
    for bb_b_one in bb_b:
        ious.append(_compute_iou(bb_a, bb_b_one))
    return max(ious), np.argmax(ious)


def gnd_bb_to_eval_bb(gnd_format):
    '''
    If gnd_format is list, return list. If it is a one value, return one value.
    gnd_format: {'x1': <value>, 'x2': <value>, 'y1': <value>, 'y2': <value>}
    return: ['x1', 'x2', 'y1', 'y2']
    '''
    return [gnd_format['x1'], gnd_format['y1'], gnd_format['x2'], gnd_format['y2']]


def gnd_bb_to_eval_bb(bb):
    return [bb['p_min']['x'], bb['p_min']['y'], bb['p_max']['x'], bb['p_max']['y']]


def convert_gnd_to_eval_format(gnd_format):
    result = defaultdict(list)
    for obj in gnd_format:
        bb = gnd_bb_to_eval_bb(obj['boundingBox'])
        result[obj['class']].append(bb)
    return result


def eval_pr_iou(testset, gnds, predicts, bb_type, dpi_val, iou_thres=0.01):
    """
    using evaluation format ({'title': [<title_bb>], 'author': [<author_bb>]})
    Each bb format: ['x1', 'y1', 'x2', 'y2']
    :param annot_dir:
    :param test_set:
    :param predicts:
    :return:
    """

    def compute_max_iou_each_predict(bb_a, bb_b, bb_type):
        """
        gnd: Noah's annotationg format
        predict: {'title': [], 'author':[]} (Use _pred_to_cat function)
        """
        # compute all iou's for each GND bb
        max_iou = []
        for bb_a_one in bb_a[bb_type]:
            max_iou.append(_compute_max_iou(bb_a_one, bb_b[bb_type]))
        return max_iou

    #
    pr_curves = []
    tp_all = []
    fp_all = []
    npred_all = 0
    npos_all = 0
    conf_all = []
    for test_sample in testset:
        # print("for image: ", test_sample)
        annot = gnds[test_sample]
        annot_eval_format = convert_gnd_to_eval_format(annot)
        # predicts_eval_format = convert_yolo_to_eval_format(predicts[test_sample])
        #predicts_eval_format = dets_to_bb(predicts[test_sample])

        predicts_eval_format = predicts[test_sample]

        # sort the predicted BB with its confidences (high to low)
        # print("predicts_eval ", predicts_eval_format)
        # print("annotation: ", annot)
        predicts_eval_format[bb_type] = sorted(predicts_eval_format[bb_type], key=lambda x: -x[4])

        # # compute iou for each gnd bb
        max_iou_for_each_predict = compute_max_iou_each_predict(predicts_eval_format,
                                                                annot_eval_format, bb_type)

        # compute PR
        tp = np.zeros(len(max_iou_for_each_predict))
        fp = np.zeros(len(max_iou_for_each_predict))
        npos = len(annot_eval_format[bb_type])  # number of positive GND
        npred = 0  # number of predictions
        used_gnd_idx = []
        for i, max_iou in enumerate(max_iou_for_each_predict):
            if max_iou[0] > iou_thres:
                if max_iou[1] not in used_gnd_idx:
                    tp[i] = 1
                    used_gnd_idx.append(max_iou[1])
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
            npred += 1
        conf = [x[4] for x in predicts_eval_format[bb_type]]
        tp_all.extend(tp)
        fp_all.extend(fp)
        conf_all.extend(conf)
        try:
            assert (npred == (tp + fp).sum())
        except:
            import pdb;
            pdb.set_trace()
        npos_all += npos
        npred_all += npred
    # sort the predicted BB with its confidences (high to low)
    idx = np.argsort(-np.array(conf_all))
    tp_all = np.array(tp_all)[idx]
    tp = np.cumsum(tp_all)
    fp_all = np.array(fp_all)[idx]
    fp = np.cumsum(fp_all)
    if npos_all == 0:
        return [0], [0], float('nan'), float('nan')
    recall = tp / npos_all
    precision = tp / np.maximum((tp + fp), 0.000001)
    # linear regression to plot continuous PR curves
    x_vals = np.arange(0, 1, 0.001)
    max_recall = 0
    if len(recall) == 0 or len(precision) == 0:
        pr_curve = np.zeros(1000)
        max_recall = 0.0
    else:
        pr_curve = np.interp(x_vals, np.array(recall), np.array(precision))
        pr_curve[int(max(recall) * 1000) + 1:] = 0  # cut off precision after max-recall
        max_recall = max(recall)
    pr_curves.append(pr_curve)
    pr_curve = np.array(pr_curves).mean(axis=0)
    return pr_curve, x_vals, pr_curve.mean(), max_recall


def visualize_pr(recall, prec, title_val, save_path):
    import matplotlib
    if not save_path == None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot(recall, prec, 'r-')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.title(title_val)
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def visualize_token(img, tokens, save_or_show='show', display_name_or_save_fn='visualize'):
    cmap = {'title': (0, 0, 255), 'author': (255, 0, 0), 'others': (0, 255, 0)}

    for token in tokens:
        bb = token['bb']
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                      color=cmap[token['label']], thickness=1)

    if save_or_show == 'show':
        cv2.namedWindow(display_name_or_save_fn)
        cv2.imshow(display_name_or_save_fn, img)
        # cv2.moveWindow(display_name, 500, 50)
        cv2.waitKey(1)
    elif save_or_show == 'save':
        cv2.imwrite(display_name_or_save_fn, img)
    else:
        print('save_or_show should be either show or save. Your value is', save_or_show)


def compute_pr_curve(config, classes, test_set, annotations, all_results, save_dir_pr_curve,
                     dpi_val, show_or_save='save'):
    # read gnd
    annot_dir = os.path.join(config['dataset_path'], 'annotations')
    gnd_annots = annotations

    iou_thres = config['iou_threshold']
    for bb_type in classes:
        for iou_thres_val in iou_thres[bb_type]:
            precision, recall, auc, max_recall = eval_pr_iou(test_set, gnd_annots, all_results,
                                                             bb_type, dpi_val, iou_thres_val)
            print(
            bb_type, ', iou threshold:', iou_thres_val, ', auc:', auc, 'max_recall:', max_recall)
            if show_or_save == 'save':  # todo: always save
                save_path = os.path.join(save_dir_pr_curve, bb_type + '_iou_thres' + str(
                    iou_thres_val) + '_pr_new.pdf')
                visualize_pr(recall, precision,
                             "'" + bb_type + "' iou_thres: " + str(iou_thres_val) + " auc: " + str(
                                 auc),
                             save_path=save_path)
            elif show_or_save == 'show':
                visualize_pr(recall, precision,
                             "'" + bb_type + "' iou_thres: " + str(iou_thres_val) + " auc: " + str(
                                 auc),
                             save_path=None)  # show
            else:
                pass


def visualize_token(img, tokens, save_or_show='show', display_name_or_save_fn='visualize'):
    cmap = {'rightangle': (0, 0, 255), 'circle': (255, 0, 0), 'triangle': (0, 255, 0),
            'label_linelength': (0, 255, 0), 'label_annotation': (100, 100, 100),
            'label_line': (0, 167, 30), 'label_point': (167, 30, 167), 'label_angle': (0, 30, 200),
            'label_circle': (200, 0, 30), 'others': (255, 255, 255)}

    for token in tokens:
        bb = token['bb']
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                      color=cmap[token['label']], thickness=1)

    if save_or_show == 'show':
        cv2.namedWindow(display_name_or_save_fn)
        cv2.imshow(display_name_or_save_fn, img)
        # cv2.moveWindow(display_name, 500, 50)
        cv2.waitKey(1)
    elif save_or_show == 'save':
        cv2.imwrite(display_name_or_save_fn, img)
    else:
        print('save_or_show should be either show or save. Your value is', save_or_show)


def is_exactly_correct(predicted_tokens, classnames):
    are_correct = {classname: True for classname in classnames}
    for token in predicted_tokens:
        are_correct[token['gnd_label']] = are_correct[token['gnd_label']] \
                                          and (token['gnd_label'] == token['label'])
    return are_correct


def find_most_overlaping_label(token_bb, predicted_bbs, iou_thres):
    if len(predicted_bbs) == 0:
        return "others", 0, 0, [0, 0, 0, 0], -1
    conf_list = []
    for predicted_bb in predicted_bbs:
        io_first = _compute_iou(token_bb, predicted_bb[1:5])
        #  print(" iou ", io_first)
        if io_first > iou_thres:
            conf_list.append(io_first)
        else:
            conf_list.append(-1.0)
    conf_array = np.array(conf_list)
    argmax_conf = np.argmax(conf_array)
    # print("win prediction ", argmax_conf, "chosen ", predicted_bbs[argmax_conf])
    return predicted_bbs[argmax_conf][0], predicted_bbs[argmax_conf][5], conf_array[argmax_conf], \
           predicted_bbs[argmax_conf][1:5], argmax_conf


def find_most_confident_label(token_bb, predicted_bbs, iou_thres):
    # compute the iou and return label of most overlapping label
    if len(predicted_bbs) == 0:
        return [0, 0, 0, 0], 0

    conf_list = []
    for predicted_bb in predicted_bbs:
        if compute_iofirst(token_bb, predicted_bb[1:5]) > iou_thres:
            conf_list.append(predicted_bb[5])
        else:
            conf_list.append(-1.0)
    conf_array = np.array(conf_list)
    argmax_conf = np.argmax(conf_array)
    return predicted_bbs[argmax_conf][0], conf_array[argmax_conf], predicted_bbs[argmax_conf][1:5]


def compute_tokenwise_eval(devkit_dir, test_set, classnames, all_results, ground_truths,
                           save_dir_viz, predicted_token_dir, dpi_val, visualize=False,
                           debug=False, iou_threshold=0.5):
    viz_output_dir = save_dir_viz + '_token_based'
    if not os.path.exists(viz_output_dir):
        os.makedirs(viz_output_dir)
    token_annot_dir = os.path.join(devkit_dir, 'token_annotations')
    token_confidence_thres = iou_threshold  # todo: hyperparam
    overlap_thresh = iou_threshold
    accuracy_matching_full_allowing_falsepos = {}
    accuracy_matching_full_exact = defaultdict(lambda: 0.0)
    for ii, test_sample in enumerate(tqdm(test_set)):
        if debug:
            print("test_sample: ", test_sample)
        gnd_tokens = ground_truths[test_sample]
        img_fn = test_sample
        predicted_bbs = all_results[img_fn]
        print("all results at {} : {}".format(img_fn, predicted_bbs))
        #predicted_bbs = convert_to_my_format_from_predicted(predicted_bbs_yolo)
        img_path_fn = os.path.join(devkit_dir, 'images', img_fn)
        img = cv2.imread(img_path_fn)

        # labeling tokens by bounding box
        predicted_tokens = []
        if debug:
            print("predicted: ", predicted_bbs)
        used_predicted = dict()
        false_positives = []
        for gnd_indx, token in enumerate(gnd_tokens):
            # finding label methods
            # predicted_label, overlap_val = find_most_overlapping_label(token['bb'], predicted_bbs)
            gnd_bb = gnd_bb_to_eval_bb(token['boundingBox'])
            if debug:
                print("gnd ", token)
            predicted_label, confidence, overlap_val, predicted_bb, predicted_index = find_most_overlaping_label(
                gnd_bb, predicted_bbs, iou_thres=iou_threshold)

            if debug:
                print(
                "most overlapping ", predicted_label, "overlap val ", overlap_val, " confidence ",
                confidence, " predicted_bb ", predicted_bb)
            if confidence < token_confidence_thres or overlap_val < overlap_thresh:
                if debug:
                    print (
                    "changing to others: ", predicted_label, " overlap ", overlap_val, " gnd_bb ",
                    gnd_bb, " predicted_bb ", predicted_bb, " gnd _ class ", token['class'])
                predicted_label = 'others'
            this_item = {'bb': predicted_bb, 'gnd_bb': gnd_bb, 'label': predicted_label,
                         'overlap': overlap_val, 'id': token['id'],
                         'gnd_label': token['class']}
            if gnd_indx not in used_predicted:
                used_predicted[gnd_indx] = this_item
            else:
                other_associated = used_predicted[gnd_indx]
                if other_associated["overlap"] < overlap_val:
                    false_positives.append(other_associated)
                    used_predicted[gnd_indx] = this_item
                else:
                    false_positives.append(this_item)

            predicted_tokens.append(this_item)

        # save predicted tokens
        dump_json_object(predicted_tokens,
                         os.path.join(predicted_token_dir, test_sample + '_pred_token.json'))

        # compare prediction and gnd to compute accuracy per token (do not compute false positive)
        accuracy_per_label = {}
        num_token_per_label = {}
        for i, token in enumerate(gnd_tokens):
            if token['class'] not in accuracy_per_label:
                accuracy_per_label[token['class']] = 0.0
            if token['class'] not in num_token_per_label:
                num_token_per_label[token['class']] = 0.0
            accuracy_per_label[token['class']] += 1 if token['class'] == predicted_tokens[i][
                'label'] else 0
            num_token_per_label[token['class']] += 1
        for label_val in accuracy_per_label:
            accuracy_per_label[label_val] /= num_token_per_label[label_val]

        # match full gnd entities in this document
        for label_val in accuracy_per_label:
            if label_val not in accuracy_matching_full_allowing_falsepos:
                accuracy_matching_full_allowing_falsepos[label_val] = 0.0
            accuracy_matching_full_allowing_falsepos[label_val] += 1.0 / float(len(test_set)) if \
            accuracy_per_label[label_val] == 1.0 else 0

        # compute exact match
        are_exactly_correct = is_exactly_correct(predicted_tokens, classnames)
        for (key, is_correct) in are_exactly_correct.iteritems():
            accuracy_matching_full_exact[key] += 1.0 / float(len(test_set)) if is_correct else 0.0

        # print the stats
        if len(test_set) / 10 > 0 and ii % int(len(test_set) / 10) == 0 or len(test_set) / 10 == 0:
            # print('@', float(ii)/float(len(test_set)) * 100, '% (allowing FalsePositives):', accuracy_matching_full_allowing_falsepos)
            print('@', float(ii) / float(len(test_set)) * 100, '% (exact):',
                  accuracy_matching_full_exact)

        gnd_transformed = []
        for token in gnd_tokens:
            t_copy = copy.deepcopy(token)
            t_copy['bb'] = gnd_bb_to_eval_bb(token['boundingBox'])
            t_copy['label'] = t_copy['class']
            gnd_transformed.append(t_copy)

        to_draw_predicted = []
        for rec in predicted_bbs:
            classname = rec[0] if rec[0] != '__background__' else 'others'
            to_draw_predicted.append({'label': classname, 'bb': rec[1:5]})
            # draw gnd and predict tokens to the paper image
        # if 'title' in accuracy_per_label:
        if visualize:
            img1 = cv2.imread(img_path_fn)
            # visualize_token(img, gnd_tokens, 'show', 'gnd')
            visualize_token(img1, to_draw_predicted, 'save',
                            os.path.join(viz_output_dir, test_sample + '_predict.png'))
            # visualize_token(img, predicted_tokens, 'show', 'predict')
            img2 = cv2.imread(img_path_fn)
            visualize_token(img2, gnd_transformed, 'save',
                            os.path.join(viz_output_dir, test_sample + '_gnd.png'))

    # print('Final accuracy (allowing FalsePositives):', accuracy_matching_full_allowing_falsepos)
    print('Final accuracy (exact):', accuracy_matching_full_exact)