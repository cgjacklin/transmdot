import numpy as np
import multiprocessing
import os
import sys
from itertools import product
from collections import OrderedDict
from lib.test.evaluation import Sequence, Tracker
import torch


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)
    if seq.dataset in ['trackingnet', 'got10k']:
        if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
            os.makedirs(os.path.join(tracker.results_dir, seq.dataset))
    '''2021.1.5 create new folder for these two datasets'''
    if seq.dataset in ['trackingnet', 'got10k']:
        base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
    else:
        base_results_path = os.path.join(tracker.results_dir, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        if key == 'max_score':    # 保存score
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_max_score.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_max_score.txt'.format(base_results_path)
                save_score(bbox_file, data)

        if key == 'APCE':    # 保存APCE
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_APCE.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving APCE...")
                bbox_file = '{}_APCE.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)

# 输出max_score
def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        # output = tracker.run_sequence(seq, debug=debug)
        output = tracker.Fuse_run_sequence(seq, debug=debug)
    else:
        try:
            # output = tracker.run_sequence(seq, debug=debug)
            output = tracker.Fuse_run_sequence(seq, debug=debug)   # 保存
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')


# moe的LaSOT数据集运行
def run_moe_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_moe_sequence(seq, tracker_info, debug=debug)
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_moe_sequence, param_list)
    print('Done')


# moe
def run_moe_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            if seq.dataset in ['trackingnet', 'got10k']:
                base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
                bbox_file = '{}.txt'.format(base_results_path)
            else:
                bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        # output = tracker.run_sequence(seq, debug=debug)
        output = tracker.moe_run_sequence(seq, debug=debug)
    else:
        try:
            # output = tracker.run_sequence(seq, debug=debug)
            output = tracker.moe_run_sequence(seq, debug=debug)   # 保存
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq, tracker, output)





# mdot数据集运行
def run_mdot_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    len_data = len(dataset) // 2
    dataset_A = dataset[:len_data]
    dataset_B = dataset[len_data:]



    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq_a, seq_b in zip(dataset_A, dataset_B):
            for tracker_info in trackers:
                run_multi_sequence(seq_a, seq_b, tracker_info, debug=debug)    # 将双机对应的sequence传入
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')



# 在多机上追踪
def run_multi_sequence(seq_a: Sequence, seq_b: Sequence , tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist_a():
        if seq_a.object_ids is None:
            bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq_a.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq_a.name, obj_id) for obj_id in seq_a.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist_a() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq_a.name))

    if debug:
        output_a, output_b = tracker.Fuse_multi_run_sequence(seq_a, seq_b, debug=debug)
    else:
        try:
            output_a, output_b = tracker.Fuse_multi_run_sequence(seq_a, seq_b, debug=debug)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output_a['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output_a['time']])
        num_frames = len(output_a['time'])
    else:
        exec_time = sum(output_a['time'])
        num_frames = len(output_a['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq_a, tracker, output_a)
        _save_tracker_output(seq_b, tracker, output_b)


# Three mdot数据集运行
def run_mdot_dataset_three(dataset, trackers, debug=False, threads=0, num_gpus=8):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """
    multiprocessing.set_start_method('spawn', force=True)

    print('Evaluating {:4d} trackers on {:5d} sequences'.format(len(trackers), len(dataset)))

    multiprocessing.set_start_method('spawn', force=True)

    len_data = len(dataset) // 3
    len_two_data = len_data * 2
    dataset_A = dataset[:len_data]
    dataset_B = dataset[len_data:len_two_data]
    dataset_C = dataset[len_two_data:]



    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq_a, seq_b, seq_c in zip(dataset_A, dataset_B, dataset_C):
            for tracker_info in trackers:
                run_three_multi_sequence(seq_a, seq_b, seq_c, tracker_info, debug=debug)    # 将双机对应的sequence传入
    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus) for seq, tracker_info in product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done')


# 在多机上追踪
def run_three_multi_sequence(seq_a: Sequence, seq_b: Sequence ,seq_c: Sequence, tracker: Tracker, debug=False, num_gpu=8):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist_a():
        if seq_a.object_ids is None:
            bbox_file = '{}/{}.txt'.format(tracker.results_dir, seq_a.name)
            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq_a.name, obj_id) for obj_id in seq_a.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist_a() and not debug:
        print('FPS: {}'.format(-1))
        return

    print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq_a.name))

    if debug:
        output_a, output_b, output_c = tracker.Fuse_three_multi_run_sequence(seq_a, seq_b, seq_c, debug=debug)
    else:
        try:
            output_a, output_b, output_c = tracker.Fuse_three_multi_run_sequence(seq_a, seq_b, seq_c, debug=debug)
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output_a['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output_a['time']])
        num_frames = len(output_a['time'])
    else:
        exec_time = sum(output_a['time'])
        num_frames = len(output_a['time'])

    print('FPS: {}'.format(num_frames / exec_time))

    if not debug:
        _save_tracker_output(seq_a, tracker, output_a)
        _save_tracker_output(seq_b, tracker, output_b)
        _save_tracker_output(seq_c, tracker, output_c)