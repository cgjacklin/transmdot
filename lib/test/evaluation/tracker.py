import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv

from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np

import copy


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids = None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')


    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")


# 融合结果
    def Fuse_run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self.Fuse_track_sequence(tracker, seq, init_info)
        return output

# 融合结果
    def Fuse_track_sequence(self, tracker, seq, init_info):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out, max_score, response_APCE = tracker.Fusetrack(image, info)              # 保存了score
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time, 'max_score': max_score.cpu(), 'APCE':response_APCE.cpu()})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output



# 融合结果
    def Fuse_multi_run_sequence(self, seq_a, seq_b, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info_a = seq_a.init_info()
        init_info_b = seq_b.init_info()

        tracker = self.create_tracker(params)
        tracker2 = self.create_tracker(params)

        #output_a, output_b = self.Fuse_multi_track_sequence(tracker, tracker2, seq_a, seq_b, init_info_a, init_info_b)     # 这边回传的两个结果还没处理
        output_a, output_b = self.Fuse_multi_track_matching_sequence(tracker, tracker2, seq_a, seq_b, init_info_a, init_info_b)       # 多机匹配候选区域
        return output_a, output_b



# 融合结果, 多模板
    def Fuse_multi_track_sequence(self, tracker,tracker2, seq_a, seq_b, init_info_a, init_info_b):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output_a = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}
        output_b = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}

        if tracker.params.save_all_boxes:
            output_a['all_boxes'] = []
            output_a['all_scores'] = []

        if tracker2.params.save_all_boxes:
            output_b['all_boxes'] = []
            output_b['all_scores'] = []

        def _store_outputs(output,tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image_a = self._read_image(seq_a.frames[0])
        image_b = self._read_image(seq_b.frames[0])

        start_time = time.time()

        # img_list = [image_a, image_b]
        # info_list = [init_info_a, init_info_b]
        out_a = tracker.multi_initialize(image_a, image_b, init_info_a, init_info_b)    ##### 改成初始化多机
        out_b = tracker2.multi_initialize(image_b, image_a, init_info_b, init_info_a)    ##### 改成初始化多机

        if out_a is None:
            out_a = {}

        if out_b is None:
            out_b = {}

        prev_output_a = OrderedDict(out_a)
        init_default_a = {'target_bbox': init_info_a.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}

        prev_output_b = OrderedDict(out_b)
        init_default_b = {'target_bbox': init_info_b.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}
                        
        if tracker.params.save_all_boxes:
            init_default_a['all_boxes'] = out_a['all_boxes']
            init_default_a['all_scores'] = out_a['all_scores']

        if tracker2.params.save_all_boxes:
            init_default_b['all_boxes'] = out_b['all_boxes']
            init_default_b['all_scores'] = out_b['all_scores']

        _store_outputs(output_a, out_a, init_default_a)

        _store_outputs(output_b, out_b, init_default_b)



        for frame_num, frame_path in enumerate(seq_a.frames[1:], start=1):
            image_a = self._read_image(frame_path)
            image_b = self._read_image(seq_b.frames[frame_num])

            start_time = time.time()

            info_a = seq_a.frame_info(frame_num)
            info_a['previous_output'] = prev_output_a

            info_b = seq_b.frame_info(frame_num)
            info_b['previous_output'] = prev_output_b

            if len(seq_a.ground_truth_rect) > 1:
                info_a['gt_bbox'] = seq_a.ground_truth_rect[frame_num]
            
            if len(seq_b.ground_truth_rect) > 1:
                info_b['gt_bbox'] = seq_b.ground_truth_rect[frame_num]

            out_a, max_score_a, response_APCE_a = tracker.multi_Fusetrack(image_a, image_b, "a", info_a, info_b)                              # 保存了score

            out_b, max_score_b, response_APCE_b = tracker2.multi_Fusetrack(image_b, image_a, "b",info_b, info_a)

            state_a = tracker.state
            state_b = tracker2.state
            
            # 双模板和双搜索区域
            # out_a, max_score_a = tracker.multi_Fusetrack2(image_a, image_b, state_b, info_a, info_b)                              # 保存了score

            # out_b, max_score_b = tracker2.multi_Fusetrack2(image_b, image_a, state_a, info_b, info_a)

            prev_output_a = OrderedDict(out_a)
            _store_outputs(output_a, out_a, {'time': time.time() - start_time, 'max_score': max_score_a.cpu(), 'APCE':response_APCE_a.cpu()})
            prev_output_b = OrderedDict(out_b)
            _store_outputs(output_b, out_b, {'time': time.time() - start_time, 'max_score': max_score_b.cpu(), 'APCE':response_APCE_b.cpu()})


        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_a and len(output_a[key]) <= 1:
                output_a.pop(key)
        
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_b and len(output_b[key]) <= 1:
                output_b.pop(key)

        return output_a, output_b




# 多机融合结果，并且双机匹配，当目标丢失时把Search region映射过去
    def Fuse_multi_track_matching_sequence(self, tracker,tracker2, seq_a, seq_b, init_info_a, init_info_b):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output_a = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}
        output_b = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}

        if tracker.params.save_all_boxes:
            output_a['all_boxes'] = []
            output_a['all_scores'] = []

        if tracker2.params.save_all_boxes:
            output_b['all_boxes'] = []
            output_b['all_scores'] = []

        def _store_outputs(output,tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image_a = self._read_image(seq_a.frames[0])
        image_b = self._read_image(seq_b.frames[0])

        start_time = time.time()

        # img_list = [image_a, image_b]
        # info_list = [init_info_a, init_info_b]
        out_a = tracker.multi_initialize(image_a, image_b, init_info_a, init_info_b)    ##### 改成初始化多机
        out_b = tracker2.multi_initialize(image_b, image_a, init_info_b, init_info_a)    ##### 改成初始化多机

        if out_a is None:
            out_a = {}

        if out_b is None:
            out_b = {}

        prev_output_a = OrderedDict(out_a)
        init_default_a = {'target_bbox': init_info_a.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}

        prev_output_b = OrderedDict(out_b)
        init_default_b = {'target_bbox': init_info_b.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}
                        
        if tracker.params.save_all_boxes:
            init_default_a['all_boxes'] = out_a['all_boxes']
            init_default_a['all_scores'] = out_a['all_scores']

        if tracker2.params.save_all_boxes:
            init_default_b['all_boxes'] = out_b['all_boxes']
            init_default_b['all_scores'] = out_b['all_scores']

        _store_outputs(output_a, out_a, init_default_a)

        _store_outputs(output_b, out_b, init_default_b)



        for frame_num, frame_path in enumerate(seq_a.frames[1:], start=1):
            image_a = self._read_image(frame_path)
            image_b = self._read_image(seq_b.frames[frame_num])

            start_time = time.time()

            info_a = seq_a.frame_info(frame_num)
            info_a['previous_output'] = prev_output_a

            info_b = seq_b.frame_info(frame_num)
            info_b['previous_output'] = prev_output_b

            if len(seq_a.ground_truth_rect) > 1:
                info_a['gt_bbox'] = seq_a.ground_truth_rect[frame_num]
            
            if len(seq_b.ground_truth_rect) > 1:
                info_b['gt_bbox'] = seq_b.ground_truth_rect[frame_num]

            out_a, max_score_a, response_APCE_a = tracker.multi_Fusetrack(image_a, image_b, "a", info_a, info_b)                              # 保存了score

            out_b, max_score_b, response_APCE_b = tracker2.multi_Fusetrack(image_b, image_a, "b", info_b, info_a)

            state_a = copy.deepcopy(tracker.state)
            state_b = copy.deepcopy(tracker2.state)

            # out_a, max_score_a = tracker.multi_Fusetrack2(image_a, image_b, state_b, info_a, info_b)                              # 保存了score

            # out_b, max_score_b = tracker2.multi_Fusetrack2(image_b, image_a, state_a, info_b, info_a)

            

            # redet_scale = [7, 12, 20]
            # if (len(output_a['max_score'])>5) and (sum(output_a['max_score'][-4:])+max_score_a.cpu())/(len(output_a['max_score'][-4:])+1) < 0.3:      # 过去5帧均值小于阈值

            #     for i, scale in enumerate(redet_scale):
            #         out_a_tmp, max_score_a_tmp, response_APCE_a_tmp = tracker.general_redetect(image_a, image_b, "a", scale, info_a, info_b)      # 逐步扩大搜索区域的重检测
            #         if max_score_a_tmp > max_score_a:
            #             print("accept redetection")
            #             out_a, max_score_a, response_APCE_a = out_a_tmp, max_score_a_tmp, response_APCE_a_tmp
            #             break
            #         tracker.state = state_a   # 状态重置

            # elif (len(output_b['max_score'])>5) and (sum(output_b['max_score'][-4:])+max_score_b.cpu())/(len(output_b['max_score'][-4:])+1) < 0.3:

            #     for i, scale in enumerate(redet_scale):
            #         out_b_tmp, max_score_b_tmp, response_APCE_b_tmp = tracker2.general_redetect(image_b, image_a, "b", scale, info_b, info_a)      # 逐步扩大搜索区域的重检测
            #         if max_score_b_tmp > max_score_b:
            #             print("accept redetection")
            #             out_b, max_score_b, response_APCE_b = out_b_tmp, max_score_b_tmp, response_APCE_b_tmp
            #             break
            #         tracker2.state = state_b   # 状态重置


            #######################################################  跨机重检测  ##########################################
            redet_factor_list = [[4,12], [3,9], [2,5]]
            # redet_factor_list = [[4,12], [3,9]]

            if((max_score_a < 0.2 and response_APCE_a < 100) and (max_score_b > 0.3) and (response_APCE_b > response_APCE_a)):            # B的置信度比A高，这之后引入不确定性的知识。 将B的Search region映射给A

                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker.state = copy.deepcopy(state_a)
                    out_a_tmp, max_score_a_tmp, response_APCE_a_tmp = tracker.search_redetect(image_a, image_b, "a", copy.deepcopy(state_b), factor[0], factor[1], info_a, info_b)
                    tmp_dict = {"out_a":out_a_tmp, "max_score_a":max_score_a_tmp, "response_APCE_a":response_APCE_a_tmp}
                    redet_results.append(tmp_dict)
                    

                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_a"] > ms:
                        ms = result_dict["max_score_a"]
                        label = i

                if redet_results[label]["max_score_a"] - max_score_a > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_a, max_score_a, response_APCE_a = redet_results[label]["out_a"], redet_results[label]["max_score_a"], redet_results[label]["response_APCE_a"]
                    tracker.state = out_a["target_bbox"]

                else:
                    print("remain ori")
                    tracker.state = copy.deepcopy(state_a)


            elif((max_score_b < 0.2 and response_APCE_b < 100) and (max_score_a > 0.3) and (response_APCE_a > response_APCE_b)):            # B的置信度比A高，这之后引入不确定性的知识。 将B的Search region映射给A

                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker2.state = copy.deepcopy(state_b)
                    out_b_tmp, max_score_b_tmp, response_APCE_b_tmp = tracker2.search_redetect(image_b, image_a, "b", copy.deepcopy(state_a), factor[0], factor[1], info_b, info_a)
                    tmp_dict = {"out_b":out_b_tmp, "max_score_b":max_score_b_tmp, "response_APCE_b":response_APCE_b_tmp}
                    redet_results.append(tmp_dict)
                    

                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_b"] > ms:
                        ms = result_dict["max_score_b"]
                        label = i

                if redet_results[label]["max_score_b"] - max_score_b > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_b, max_score_b, response_APCE_b = redet_results[label]["out_b"], redet_results[label]["max_score_b"], redet_results[label]["response_APCE_b"]
                    tracker2.state = out_b["target_bbox"]

                else:
                    print("remain ori")
                    tracker2.state = copy.deepcopy(state_b)



            #双阶段double重检测
            # if((max_score_a < 0.2 and response_APCE_a < 100) and (max_score_b > 0.3) and (response_APCE_b > response_APCE_a)):            # B的置信度比A高，这之后引入不确定性的知识。 将B的Search region映射给A
            #     #out_a, max_score_a, response_APCE_a = tracker.search_redetect(image_a, image_b, "a", state_b, 4.0, 12.0, info_a, info_b)
            #     out_a_re, max_score_a_re, response_APCE_a_re = tracker.search_redetect(image_a, image_b, "a", state_b, 4.0, 12.0, info_a, info_b)
            #     # if (max_score_a < 0.4) or (response_APCE_a < 100):                # 二次重检测
            #     #     out_a_re, max_score_a_re, response_APCE_a_re = tracker.search_redetect(image_a, image_b, "a", tracker2.state, 3.0, 9.0, info_a, info_b)

            #     #if (max_score_a_re - max_score_a) > 0.1 and (response_APCE_a_re - response_APCE_a) > 0:     # 接受重检测结果
            #     #if (max_score_a_re - max_score_a) > -1:
            #     if max_score_a_re> 0:     
            #         print("accept redetection")
            #         out_a, max_score_a, response_APCE_a = out_a_re, max_score_a_re, response_APCE_a_re
            #     else:
            #         tracker.state = state_a
            #         print("remain ori state")

            # elif((max_score_b < 0.2 and response_APCE_b < 100) and (max_score_a > 0.3) and (response_APCE_a > response_APCE_b)):
            #     #out_b, max_score_b, response_APCE_b = tracker2.search_redetect(image_b, image_a, "b", state_a, 4.0, 12.0, info_b, info_a)
            #     out_b_re, max_score_b_re, response_APCE_b_re = tracker2.search_redetect(image_b, image_a, "b", state_a, 4.0, 12.0, info_b, info_a)
            #     # if (max_score_b < 0.4) or (response_APCE_b < 200):
            #     #     out_b_re, max_score_b_re, response_APCE_b_re = tracker2.search_redetect(image_b, image_a, "b", tracker.state, 3.0, 9.0, info_b, info_a)

            #     #if (max_score_b_re - max_score_b) > 0.1 and (response_APCE_b_re - response_APCE_b) > 0:     # 接受重检测结果
            #     #if (max_score_b_re - max_score_b) > -1:
            #     if max_score_b_re> 0:  
            #         print("accept redetection")
            #         out_b, max_score_b, response_APCE_b = out_b_re, max_score_b_re, response_APCE_b_re
            #     else:
            #         tracker2.state = state_b
            #         print("remain ori state")


            prev_output_a = OrderedDict(out_a)
            _store_outputs(output_a, out_a, {'time': time.time() - start_time, 'max_score': max_score_a.cpu(),  'APCE':response_APCE_a.cpu()})
            prev_output_b = OrderedDict(out_b)
            _store_outputs(output_b, out_b, {'time': time.time() - start_time, 'max_score': max_score_b.cpu(), 'APCE':response_APCE_b.cpu()})


        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_a and len(output_a[key]) <= 1:
                output_a.pop(key)
        
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_b and len(output_b[key]) <= 1:
                output_b.pop(key)

        return output_a, output_b



# 三机融合结果
    def Fuse_three_multi_run_sequence(self, seq_a, seq_b, seq_c, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info_a = seq_a.init_info()
        init_info_b = seq_b.init_info()
        init_info_c = seq_c.init_info()

        tracker = self.create_tracker(params)
        tracker2 = self.create_tracker(params)
        tracker3 = self.create_tracker(params)

        #output_a, output_b = self.Fuse_multi_track_sequence(tracker, tracker2, seq_a, seq_b, init_info_a, init_info_b)     # 这边回传的两个结果还没处理
        output_a, output_b, output_c = self.Fuse_three_multi_track(tracker, tracker2, tracker3, seq_a, seq_b,seq_c, init_info_a, init_info_b,init_info_c)      
        #output_a, output_b, output_c = self.Fuse_three_multi_track_matching_sequence(tracker, tracker2, tracker3, seq_a, seq_b,seq_c, init_info_a, init_info_b,init_info_c)       # 多机匹配候选区域
        return output_a, output_b,output_c


# 三机融合结果，三模板无重检测
    def Fuse_three_multi_track(self, tracker, tracker2, tracker3, seq_a, seq_b, seq_c, init_info_a, init_info_b,init_info_c):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output_a = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}
        output_b = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}
        output_c = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}

        if tracker.params.save_all_boxes:
            output_a['all_boxes'] = []
            output_a['all_scores'] = []

        if tracker2.params.save_all_boxes:
            output_b['all_boxes'] = []
            output_b['all_scores'] = []

        if tracker3.params.save_all_boxes:
            output_c['all_boxes'] = []
            output_c['all_scores'] = []

        def _store_outputs(output,tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image_a = self._read_image(seq_a.frames[0])
        image_b = self._read_image(seq_b.frames[0])
        image_c = self._read_image(seq_c.frames[0])

        start_time = time.time()

        # img_list = [image_a, image_b]
        # info_list = [init_info_a, init_info_b]
        out_a = tracker.three_multi_initialize(image_a, image_b, image_c, init_info_a, init_info_b, init_info_c)    ##### 改成初始化多机
        out_b = tracker2.three_multi_initialize(image_b, image_a, image_c, init_info_b, init_info_a, init_info_c)    ##### 改成初始化多机
        out_c = tracker3.three_multi_initialize(image_c, image_a, image_b, init_info_c, init_info_a, init_info_b)    ##### 改成初始化多机

        if out_a is None:
            out_a = {}

        if out_b is None:
            out_b = {}

        if out_c is None:
            out_c = {}

        prev_output_a = OrderedDict(out_a)
        init_default_a = {'target_bbox': init_info_a.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}

        prev_output_b = OrderedDict(out_b)
        init_default_b = {'target_bbox': init_info_b.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}

        prev_output_c = OrderedDict(out_c)
        init_default_c = {'target_bbox': init_info_c.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}
                        

        if tracker.params.save_all_boxes:
            init_default_a['all_boxes'] = out_a['all_boxes']
            init_default_a['all_scores'] = out_a['all_scores']

        if tracker2.params.save_all_boxes:
            init_default_b['all_boxes'] = out_b['all_boxes']
            init_default_b['all_scores'] = out_b['all_scores']
        
        if tracker3.params.save_all_boxes:
            init_default_c['all_boxes'] = out_c['all_boxes']
            init_default_c['all_scores'] = out_c['all_scores']

        _store_outputs(output_a, out_a, init_default_a)

        _store_outputs(output_b, out_b, init_default_b)

        _store_outputs(output_c, out_c, init_default_c)



        for frame_num, frame_path in enumerate(seq_a.frames[1:], start=1):
            image_a = self._read_image(frame_path)
            image_b = self._read_image(seq_b.frames[frame_num])
            image_c = self._read_image(seq_c.frames[frame_num])

            start_time = time.time()

            info_a = seq_a.frame_info(frame_num)
            info_a['previous_output'] = prev_output_a

            info_b = seq_b.frame_info(frame_num)
            info_b['previous_output'] = prev_output_b

            info_c = seq_c.frame_info(frame_num)
            info_c['previous_output'] = prev_output_c

            if len(seq_a.ground_truth_rect) > 1:
                info_a['gt_bbox'] = seq_a.ground_truth_rect[frame_num]
            
            if len(seq_b.ground_truth_rect) > 1:
                info_b['gt_bbox'] = seq_b.ground_truth_rect[frame_num]

            if len(seq_c.ground_truth_rect) > 1:
                info_c['gt_bbox'] = seq_c.ground_truth_rect[frame_num]

            out_a, max_score_a, response_APCE_a = tracker.three_nomulti_Fusetrack(image_a, image_b, image_c, "a", info_a, info_b, info_c)                              # 保存了score

            out_b, max_score_b, response_APCE_b = tracker2.three_nomulti_Fusetrack(image_b, image_a, image_c, "b", info_b, info_a, info_c)

            out_c, max_score_c, response_APCE_c = tracker3.three_nomulti_Fusetrack(image_c, image_a, image_b, "c", info_c, info_a, info_b)



            prev_output_a = OrderedDict(out_a)
            _store_outputs(output_a, out_a, {'time': time.time() - start_time, 'max_score': max_score_a.cpu(),  'APCE':response_APCE_a.cpu()})
            prev_output_b = OrderedDict(out_b)
            _store_outputs(output_b, out_b, {'time': time.time() - start_time, 'max_score': max_score_b.cpu(), 'APCE':response_APCE_b.cpu()})
            prev_output_c = OrderedDict(out_c)
            _store_outputs(output_c, out_c, {'time': time.time() - start_time, 'max_score': max_score_c.cpu(), 'APCE':response_APCE_c.cpu()})


        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_a and len(output_a[key]) <= 1:
                output_a.pop(key)
        
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_b and len(output_b[key]) <= 1:
                output_b.pop(key)

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_c and len(output_c[key]) <= 1:
                output_c.pop(key)

        return output_a, output_b, output_c



# 三机融合结果，并且三机匹配，当目标丢失时把Search region映射过去
    def Fuse_three_multi_track_matching_sequence(self, tracker, tracker2, tracker3, seq_a, seq_b, seq_c, init_info_a, init_info_b,init_info_c):
        # Define outputs
        # Each field in output is a list containing tracker prediction for each frame.

        # In case of single object tracking mode:
        # target_bbox[i] is the predicted bounding box for frame i
        # time[i] is the processing time for frame i

        # In case of multi object tracking mode:
        # target_bbox[i] is an OrderedDict, where target_bbox[i][obj_id] is the predicted box for target obj_id in
        # frame i
        # time[i] is either the processing time for frame i, or an OrderedDict containing processing times for each
        # object in frame i

        output_a = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}
        output_b = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}
        output_c = {'target_bbox': [],
                  'time': [],
                  'max_score': [],
                  'APCE':[]}

        if tracker.params.save_all_boxes:
            output_a['all_boxes'] = []
            output_a['all_scores'] = []

        if tracker2.params.save_all_boxes:
            output_b['all_boxes'] = []
            output_b['all_scores'] = []

        if tracker3.params.save_all_boxes:
            output_c['all_boxes'] = []
            output_c['all_scores'] = []

        def _store_outputs(output,tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image_a = self._read_image(seq_a.frames[0])
        image_b = self._read_image(seq_b.frames[0])
        image_c = self._read_image(seq_c.frames[0])

        start_time = time.time()

        # img_list = [image_a, image_b]
        # info_list = [init_info_a, init_info_b]
        out_a = tracker.three_multi_initialize(image_a, image_b, image_c, init_info_a, init_info_b, init_info_c)    ##### 改成初始化多机
        out_b = tracker2.three_multi_initialize(image_b, image_a, image_c, init_info_b, init_info_a, init_info_c)    ##### 改成初始化多机
        out_c = tracker3.three_multi_initialize(image_c, image_a, image_b, init_info_c, init_info_a, init_info_b)    ##### 改成初始化多机

        if out_a is None:
            out_a = {}

        if out_b is None:
            out_b = {}

        if out_c is None:
            out_c = {}

        prev_output_a = OrderedDict(out_a)
        init_default_a = {'target_bbox': init_info_a.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}

        prev_output_b = OrderedDict(out_b)
        init_default_b = {'target_bbox': init_info_b.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}

        prev_output_c = OrderedDict(out_c)
        init_default_c = {'target_bbox': init_info_c.get('init_bbox'),
                        'time': time.time() - start_time,
                        'max_score': 0,
                        'APCE':0}
                        

        if tracker.params.save_all_boxes:
            init_default_a['all_boxes'] = out_a['all_boxes']
            init_default_a['all_scores'] = out_a['all_scores']

        if tracker2.params.save_all_boxes:
            init_default_b['all_boxes'] = out_b['all_boxes']
            init_default_b['all_scores'] = out_b['all_scores']
        
        if tracker3.params.save_all_boxes:
            init_default_c['all_boxes'] = out_c['all_boxes']
            init_default_c['all_scores'] = out_c['all_scores']

        _store_outputs(output_a, out_a, init_default_a)

        _store_outputs(output_b, out_b, init_default_b)

        _store_outputs(output_c, out_c, init_default_c)



        for frame_num, frame_path in enumerate(seq_a.frames[1:], start=1):
            image_a = self._read_image(frame_path)
            image_b = self._read_image(seq_b.frames[frame_num])
            image_c = self._read_image(seq_c.frames[frame_num])

            start_time = time.time()

            info_a = seq_a.frame_info(frame_num)
            info_a['previous_output'] = prev_output_a

            info_b = seq_b.frame_info(frame_num)
            info_b['previous_output'] = prev_output_b

            info_c = seq_c.frame_info(frame_num)
            info_c['previous_output'] = prev_output_c

            if len(seq_a.ground_truth_rect) > 1:
                info_a['gt_bbox'] = seq_a.ground_truth_rect[frame_num]
            
            if len(seq_b.ground_truth_rect) > 1:
                info_b['gt_bbox'] = seq_b.ground_truth_rect[frame_num]

            if len(seq_c.ground_truth_rect) > 1:
                info_c['gt_bbox'] = seq_c.ground_truth_rect[frame_num]

            out_a, max_score_a, response_APCE_a = tracker.three_multi_Fusetrack(image_a, image_b, image_c, "a", info_a, info_b, info_c)                              # 保存了score

            out_b, max_score_b, response_APCE_b = tracker2.three_multi_Fusetrack(image_b, image_a, image_c, "b", info_b, info_a, info_c)

            out_c, max_score_c, response_APCE_c = tracker3.three_multi_Fusetrack(image_c, image_a, image_b, "c", info_c, info_a, info_b)

            state_a = copy.deepcopy(tracker.state)
            state_b = copy.deepcopy(tracker2.state)
            state_c = copy.deepcopy(tracker3.state)

            #######################################################  跨机重检测  ##########################################
            redet_factor_list = [[4,12], [3,9], [2,5]]
            # redet_factor_list = [[4,12], [3,9]]

            if max_score_a > max(max_score_b, max_score_c):
                tmp_max_score = copy.deepcopy(max_score_a)
                tmp_APEC = copy.deepcopy(response_APCE_a)
                tmp_image = copy.deepcopy(image_a)
                tmp_state = copy.deepcopy(state_a)
                tmp_info = copy.deepcopy(info_a)
            elif max_score_b > max(max_score_a, max_score_c):
                tmp_max_score = copy.deepcopy(max_score_b)
                tmp_APEC = copy.deepcopy(response_APCE_b)
                tmp_image = copy.deepcopy(image_b)
                tmp_state = copy.deepcopy(state_b)
                tmp_info = copy.deepcopy(info_b)
            else:
                tmp_max_score = copy.deepcopy(max_score_c)
                tmp_APEC = copy.deepcopy(response_APCE_c)
                tmp_image = copy.deepcopy(image_c)
                tmp_state = copy.deepcopy(state_c)
                tmp_info = copy.deepcopy(info_c)

            if((max_score_a < 0.2 and response_APCE_a < 100) and (tmp_max_score > 0.3) and (tmp_APEC > response_APCE_a)):            # B的置信度比A高，这之后引入不确定性的知识。 将B的Search region映射给A

                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker.state = copy.deepcopy(state_a)
                    out_a_tmp, max_score_a_tmp, response_APCE_a_tmp = tracker.three_search_redetect(image_a, tmp_image, "a", copy.deepcopy(tmp_state), factor[0], factor[1], info_a, tmp_info)
                    tmp_dict = {"out_a":out_a_tmp, "max_score_a":max_score_a_tmp, "response_APCE_a":response_APCE_a_tmp}
                    redet_results.append(tmp_dict)
                    

                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_a"] > ms:
                        ms = result_dict["max_score_a"]
                        label = i

                if redet_results[label]["max_score_a"] - max_score_a > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_a, max_score_a, response_APCE_a = redet_results[label]["out_a"], redet_results[label]["max_score_a"], redet_results[label]["response_APCE_a"]
                    tracker.state = out_a["target_bbox"]

                else:
                    print("remain ori")
                    tracker.state = copy.deepcopy(state_a)


            if((max_score_b < 0.2 and response_APCE_b < 100) and (tmp_max_score > 0.3) and (tmp_APEC > response_APCE_b)):            # B的置信度比A高，这之后引入不确定性的知识。 将B的Search region映射给A

                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker2.state = copy.deepcopy(state_b)
                    out_b_tmp, max_score_b_tmp, response_APCE_b_tmp = tracker2.three_search_redetect(image_b, tmp_image, "b", copy.deepcopy(tmp_state), factor[0], factor[1], info_b, tmp_info)
                    tmp_dict = {"out_b":out_b_tmp, "max_score_b":max_score_b_tmp, "response_APCE_b":response_APCE_b_tmp}
                    redet_results.append(tmp_dict)
                    

                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_b"] > ms:
                        ms = result_dict["max_score_b"]
                        label = i

                if redet_results[label]["max_score_b"] - max_score_b > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_b, max_score_b, response_APCE_b = redet_results[label]["out_b"], redet_results[label]["max_score_b"], redet_results[label]["response_APCE_b"]
                    tracker2.state = out_b["target_bbox"]

                else:
                    print("remain ori")
                    tracker2.state = copy.deepcopy(state_b)

            if((max_score_c < 0.2 and response_APCE_c < 100) and (tmp_max_score > 0.3) and (tmp_APEC > response_APCE_c)):            # B的置信度比A高，这之后引入不确定性的知识。 将B的Search region映射给A

                redet_results = []
                for i, factor in enumerate(redet_factor_list):
                    tracker3.state = copy.deepcopy(state_c)
                    out_c_tmp, max_score_c_tmp, response_APCE_c_tmp = tracker3.three_search_redetect(image_c, tmp_image, "c", copy.deepcopy(tmp_state), factor[0], factor[1], info_c, tmp_info)
                    tmp_dict = {"out_c":out_c_tmp, "max_score_c":max_score_c_tmp, "response_APCE_c":response_APCE_c_tmp}
                    redet_results.append(tmp_dict)
                    

                label = 0
                ms = 0
                for i, result_dict in enumerate(redet_results):
                    if result_dict["max_score_c"] > ms:
                        ms = result_dict["max_score_c"]
                        label = i

                if redet_results[label]["max_score_c"] - max_score_c > 0:
                    print("used_factor:", redet_factor_list[label])
                    out_c, max_score_c, response_APCE_c = redet_results[label]["out_c"], redet_results[label]["max_score_c"], redet_results[label]["response_APCE_c"]
                    tracker3.state = out_c["target_bbox"]

                else:
                    print("remain ori")
                    tracker3.state = copy.deepcopy(state_c)

            prev_output_a = OrderedDict(out_a)
            _store_outputs(output_a, out_a, {'time': time.time() - start_time, 'max_score': max_score_a.cpu(),  'APCE':response_APCE_a.cpu()})
            prev_output_b = OrderedDict(out_b)
            _store_outputs(output_b, out_b, {'time': time.time() - start_time, 'max_score': max_score_b.cpu(), 'APCE':response_APCE_b.cpu()})
            prev_output_c = OrderedDict(out_c)
            _store_outputs(output_c, out_c, {'time': time.time() - start_time, 'max_score': max_score_c.cpu(), 'APCE':response_APCE_c.cpu()})


        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_a and len(output_a[key]) <= 1:
                output_a.pop(key)
        
        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_b and len(output_b[key]) <= 1:
                output_b.pop(key)

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output_c and len(output_c[key]) <= 1:
                output_c.pop(key)

        return output_a, output_b, output_c

    def moe_run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self.moe_track_sequence(tracker, seq, init_info)
        return output


    def moe_track_sequence(self, tracker, seq, init_info):

        output = {'target_bbox': [],
                'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output