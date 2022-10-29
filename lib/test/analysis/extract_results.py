import os
import sys
import numpy as np
from lib.test.utils.load_text import load_text
import torch
import pickle
from tqdm import tqdm

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.environment import env_settings


def calc_err_center(pred_bb, anno_bb, normalized=False):
    pred_center = pred_bb[:, :2] + 0.5 * (pred_bb[:, 2:] - 1.0)
    anno_center = anno_bb[:, :2] + 0.5 * (anno_bb[:, 2:] - 1.0)

    if normalized:
        pred_center = pred_center / anno_bb[:, 2:]
        anno_center = anno_center / anno_bb[:, 2:]

    err_center = ((pred_center - anno_center)**2).sum(1).sqrt()
    return err_center


def calc_iou_overlap(pred_bb, anno_bb):
    tl = torch.max(pred_bb[:, :2], anno_bb[:, :2])
    br = torch.min(pred_bb[:, :2] + pred_bb[:, 2:] - 1.0, anno_bb[:, :2] + anno_bb[:, 2:] - 1.0)
    sz = (br - tl + 1.0).clamp(0)

    # Area
    intersection = sz.prod(dim=1)
    union = pred_bb[:, 2:].prod(dim=1) + anno_bb[:, 2:].prod(dim=1) - intersection

    return intersection / union


def calc_seq_err_robust(pred_bb, anno_bb, dataset, target_visible=None):
    pred_bb = pred_bb.clone()

    # Check if invalid values are present
    if torch.isnan(pred_bb).any() or (pred_bb[:, 2:] < 0.0).any():
        raise Exception('Error: Invalid results')

    if torch.isnan(anno_bb).any():
        if dataset == 'uav':
            pass
        else:
            raise Exception('Warning: NaNs in annotation')

    if (pred_bb[:, 2:] == 0.0).any():
        for i in range(1, pred_bb.shape[0]):
            if (pred_bb[i, 2:] == 0.0).any() and not torch.isnan(anno_bb[i, :]).any():
                pred_bb[i, :] = pred_bb[i-1, :]

    if pred_bb.shape[0] != anno_bb.shape[0]:
        if dataset == 'lasot':
            if pred_bb.shape[0] > anno_bb.shape[0]:
                # For monkey-17, there is a mismatch for some trackers.
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                raise Exception('Mis-match in tracker prediction and GT lengths')
        else:
            # print('Warning: Mis-match in tracker prediction and GT lengths')
            if pred_bb.shape[0] > anno_bb.shape[0]:
                pred_bb = pred_bb[:anno_bb.shape[0], :]
            else:
                pad = torch.zeros((anno_bb.shape[0] - pred_bb.shape[0], 4)).type_as(pred_bb)
                pred_bb = torch.cat((pred_bb, pad), dim=0)

    pred_bb[0, :] = anno_bb[0, :]

    if target_visible is not None:
        target_visible = target_visible.bool()
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible                  # 有框且没被遮挡的
    else:
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    # handle invalid anno cases
    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot':
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid


def extract_results(trackers, dataset, report_name, skip_missing_seq=False, plot_bin_gap=0.05,
                    exclude_invalid_frames=False):
    settings = env_settings()
    eps = 1e-16

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    avg_overlap_all = torch.zeros((len(dataset), len(trackers)), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((len(dataset), len(trackers), threshold_set_overlap.numel()),
                                                dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                               dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((len(dataset), len(trackers), threshold_set_center.numel()),
                                                    dtype=torch.float32)

    valid_sequence = torch.ones(len(dataset), dtype=torch.uint8)

    for seq_id, seq in enumerate(tqdm(dataset)):
        # Load anno
        anno_bb = torch.tensor(seq.ground_truth_rect)
        target_visible = torch.tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
        for trk_id, trk in enumerate(trackers):
            # Load results
            base_results_path = '{}/{}'.format(trk.results_dir, seq.name)
            results_path = '{}.txt'.format(base_results_path)

            if os.path.isfile(results_path):
                pred_bb = torch.tensor(load_text(str(results_path), delimiter=('\t', ','), dtype=np.float64))
            else:
                if skip_missing_seq:
                    valid_sequence[seq_id] = 0
                    break
                else:
                    raise Exception('Result not found. {}'.format(results_path))

            # Calculate measures
            err_overlap, err_center, err_center_normalized, valid_frame = calc_seq_err_robust(
                pred_bb, anno_bb, seq.dataset, target_visible)

            avg_overlap_all[seq_id, trk_id] = err_overlap[valid_frame].mean()

            if exclude_invalid_frames:
                seq_length = valid_frame.long().sum()
            else:
                seq_length = anno_bb.shape[0]

            if seq_length <= 0:
                raise Exception('Seq length zero')

            ave_success_rate_plot_overlap[seq_id, trk_id, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center[seq_id, trk_id, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center_norm[seq_id, trk_id, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length

    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    # Prepare dictionary for saving data
    seq_names = [s.name for s in dataset]
    tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                     for t in trackers]

    eval_data = {'sequences': seq_names, 'trackers': tracker_names,
                 'valid_sequence': valid_sequence.tolist(),
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                 'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                 'avg_overlap_all': avg_overlap_all.tolist(),
                 'threshold_set_overlap': threshold_set_overlap.tolist(),
                 'threshold_set_center': threshold_set_center.tolist(),
                 'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data


# 融合多机结果
def fuse_extract_results(trackers, dataset, report_name, skip_missing_seq=False, plot_bin_gap=0.05,
                    exclude_invalid_frames=False):
    settings = env_settings()
    eps = 1e-16

    result_plot_path = os.path.join(settings.result_plot_path, report_name)

    if not os.path.exists(result_plot_path):
        os.makedirs(result_plot_path)

    fuse_len = int(len(dataset)/2)

    # 初始化列表
    threshold_set_overlap = torch.arange(0.0, 1.0 + plot_bin_gap, plot_bin_gap, dtype=torch.float64)
    threshold_set_center = torch.arange(0, 51, dtype=torch.float64)
    threshold_set_center_norm = torch.arange(0, 51, dtype=torch.float64) / 100.0

    avg_overlap_all = torch.zeros((fuse_len, len(trackers)), dtype=torch.float64)
    ave_success_rate_plot_overlap = torch.zeros((fuse_len, len(trackers), threshold_set_overlap.numel()),
                                                dtype=torch.float32)
    ave_success_rate_plot_center = torch.zeros((fuse_len, len(trackers), threshold_set_center.numel()),
                                               dtype=torch.float32)
    ave_success_rate_plot_center_norm = torch.zeros((fuse_len, len(trackers), threshold_set_center.numel()),
                                                    dtype=torch.float32)

    valid_sequence = torch.ones(fuse_len, dtype=torch.uint8)

    dataset_A = dataset[:fuse_len]
    dataset_B = dataset[fuse_len:]


    for seq_id, seq in enumerate(tqdm(dataset_A)):
        # Load anno
        anno_bb_a = torch.tensor(seq.ground_truth_rect)
        anno_bb_b = torch.tensor(dataset_B[seq_id].ground_truth_rect)

        target_visible_a = torch.tensor(seq.target_visible, dtype=torch.uint8) if seq.target_visible is not None else None
        target_visible_b = torch.tensor(dataset_B[seq_id].target_visible, dtype=torch.uint8) if dataset_B[seq_id].target_visible is not None else None

        for trk_id, trk in enumerate(trackers):
            # Load results
            base_results_path_a = '{}/{}'.format(trk.results_dir, seq.name)
            base_results_path_b = '{}/{}'.format(trk.results_dir, dataset_B[seq_id].name)

            results_path_a = '{}.txt'.format(base_results_path_a)
            results_path_b = '{}.txt'.format(base_results_path_b)

            score_path_a = '{}_max_score.txt'.format(base_results_path_a)      # score的txt文件
            score_path_b = '{}_max_score.txt'.format(base_results_path_b)

            APCE_path_a = '{}_APCE.txt'.format(base_results_path_a)      # score的txt文件
            APCE_path_b = '{}_APCE.txt'.format(base_results_path_b)

            if os.path.isfile(results_path_a) and os.path.isfile(results_path_b):
                pred_bb_a = torch.tensor(load_text(str(results_path_a), delimiter=('\t', ','), dtype=np.float64))
                pred_bb_b = torch.tensor(load_text(str(results_path_b), delimiter=('\t', ','), dtype=np.float64))
                score_a = np.loadtxt(str(score_path_a), dtype=np.float64)
                score_b = np.loadtxt(str(score_path_b), dtype=np.float64)
                

            else:
                if skip_missing_seq:
                    valid_sequence[seq_id] = 0
                    break
                else:
                    raise Exception('Result not found. {}'.format(results_path_a))

            if os.path.isfile(APCE_path_a):
                APEC_a = np.loadtxt(str(APCE_path_a), dtype=np.float64)
                APEC_b = np.loadtxt(str(APCE_path_b), dtype=np.float64)

                print("max_score和APEC结合")
                err_overlap, err_center, err_center_normalized, valid_frame = fuse_calc_seq_err_robust_APEC(
                    pred_bb_a,pred_bb_b, anno_bb_a, anno_bb_b, seq.dataset, target_visible_a, target_visible_b, score_a, score_b, APEC_a, APEC_b)

            else:
            # Calculate measures，继续从这算
                print("单项融合结果")
                err_overlap, err_center, err_center_normalized, valid_frame = fuse_calc_seq_err_robust(
                    pred_bb_a,pred_bb_b, anno_bb_a, anno_bb_b, seq.dataset, target_visible_a, target_visible_b, score_a, score_b)

            avg_overlap_all[seq_id, trk_id] = err_overlap[valid_frame].mean()

            if exclude_invalid_frames:
                seq_length = valid_frame.long().sum()
            else:
                seq_length = anno_bb_a.shape[0]

            if seq_length <= 0:
                raise Exception('Seq length zero')

            ave_success_rate_plot_overlap[seq_id, trk_id, :] = (err_overlap.view(-1, 1) > threshold_set_overlap.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center[seq_id, trk_id, :] = (err_center.view(-1, 1) <= threshold_set_center.view(1, -1)).sum(0).float() / seq_length
            ave_success_rate_plot_center_norm[seq_id, trk_id, :] = (err_center_normalized.view(-1, 1) <= threshold_set_center_norm.view(1, -1)).sum(0).float() / seq_length

    print('\n\nComputed results over {} / {} sequences'.format(valid_sequence.long().sum().item(), valid_sequence.shape[0]))

    # Prepare dictionary for saving data
    seq_names = [s.name for s in dataset]
    tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 'disp_name': t.display_name}
                     for t in trackers]

    eval_data = {'sequences': seq_names, 'trackers': tracker_names,
                 'valid_sequence': valid_sequence.tolist(),
                 'ave_success_rate_plot_overlap': ave_success_rate_plot_overlap.tolist(),
                 'ave_success_rate_plot_center': ave_success_rate_plot_center.tolist(),
                 'ave_success_rate_plot_center_norm': ave_success_rate_plot_center_norm.tolist(),
                 'avg_overlap_all': avg_overlap_all.tolist(),
                 'threshold_set_overlap': threshold_set_overlap.tolist(),
                 'threshold_set_center': threshold_set_center.tolist(),
                 'threshold_set_center_norm': threshold_set_center_norm.tolist()}

    with open(result_plot_path + '/eval_data.pkl', 'wb') as fh:
        pickle.dump(eval_data, fh)

    return eval_data



def fuse_calc_seq_err_robust(pred_bb_a, pred_bb_b, anno_bb_a, anno_bb_b, dataset, target_visible_a, target_visible_b, score_a, score_b):

    pred_bb_a = pred_bb_a.clone()
    pred_bb_b = pred_bb_b.clone()


    pred_bb_a[0, :] = anno_bb_a[0, :]
    pred_bb_b[0, :] = anno_bb_b[0, :]


    # 计算置信度高的
    fused_index = []
    for i, (s_a, s_b) in enumerate(zip(score_a, score_b)):
        if s_a >= s_b:
            fused_index.append(0)
        else:
            fused_index.append(1)

    fused_index = torch.tensor(fused_index)

    pred_bb = torch.mul(pred_bb_a, torch.unsqueeze((-fused_index + 1), dim=1)) + torch.mul(pred_bb_b, torch.unsqueeze(fused_index,dim=1))
    anno_bb = torch.mul(anno_bb_a, torch.unsqueeze((-fused_index + 1), dim=1)) + torch.mul(anno_bb_b, torch.unsqueeze(fused_index,dim=1))
    
    target_visible = torch.ones(len(fused_index), dtype=int)

    for i, drone_index in enumerate(fused_index):
        if drone_index == 0:
            target_visible[i] = target_visible_a[i]
        elif drone_index == 1:
            target_visible[i] = target_visible_b[i]

    target_visible = torch.tensor(target_visible)

    if target_visible is not None:
        target_visible = target_visible.bool()
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    # handle invalid anno cases
    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot':
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid



def fuse_calc_seq_err_robust_APEC(pred_bb_a, pred_bb_b, anno_bb_a, anno_bb_b, dataset, target_visible_a, target_visible_b, score_a, score_b, APEC_a, APEC_b):

    pred_bb_a = pred_bb_a.clone()
    pred_bb_b = pred_bb_b.clone()


    pred_bb_a[0, :] = anno_bb_a[0, :]
    pred_bb_b[0, :] = anno_bb_b[0, :]


    # 计算置信度高的
    fused_index = []
    for i, (s_a, s_b, A_a, A_b) in enumerate(zip(score_a, score_b, APEC_a, APEC_b)):

        if (s_a<0.3) and (s_b<0.3):
            if A_a >= A_b:
                fused_index.append(0)
            else:
                fused_index.append(1)
        elif s_a >= s_b:
            fused_index.append(0)
        else:
            fused_index.append(1)

    fused_index = torch.tensor(fused_index)

    pred_bb = torch.mul(pred_bb_a, torch.unsqueeze((-fused_index + 1), dim=1)) + torch.mul(pred_bb_b, torch.unsqueeze(fused_index,dim=1))
    anno_bb = torch.mul(anno_bb_a, torch.unsqueeze((-fused_index + 1), dim=1)) + torch.mul(anno_bb_b, torch.unsqueeze(fused_index,dim=1))
    
    target_visible = torch.ones(len(fused_index), dtype=int)

    for i, drone_index in enumerate(fused_index):
        if drone_index == 0:
            target_visible[i] = target_visible_a[i]
        elif drone_index == 1:
            target_visible[i] = target_visible_b[i]

    target_visible = torch.tensor(target_visible)

    if target_visible is not None:
        target_visible = target_visible.bool()
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2) & target_visible
    else:
        valid = ((anno_bb[:, 2:] > 0.0).sum(1) == 2)

    err_center = calc_err_center(pred_bb, anno_bb)
    err_center_normalized = calc_err_center(pred_bb, anno_bb, normalized=True)
    err_overlap = calc_iou_overlap(pred_bb, anno_bb)

    # handle invalid anno cases
    if dataset in ['uav']:
        err_center[~valid] = -1.0
    else:
        err_center[~valid] = float("Inf")
    err_center_normalized[~valid] = -1.0
    err_overlap[~valid] = -1.0

    if dataset == 'lasot':
        err_center_normalized[~target_visible] = float("Inf")
        err_center[~target_visible] = float("Inf")

    if torch.isnan(err_overlap).any():
        raise Exception('Nans in calculated overlap')
    return err_overlap, err_center, err_center_normalized, valid