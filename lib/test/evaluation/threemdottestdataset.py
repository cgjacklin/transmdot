import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class ThreemdotDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from http://vision.cs.stonybrook.edu/~lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.threemdot_test_path
        self.sequence_list = self._get_sequence_list()
        self.clean_list = self.clean_seq_list()

    def clean_seq_list(self):
        clean_lst = []
        for i in range(len(self.sequence_list)):
            cls, _ = self.sequence_list[i].split('-')
            clean_lst.append(cls)
        return  clean_lst

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name.split('-')[0]
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        # occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)
        occlusion_label_path = '{}/{}/{}/occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name
        return Sequence(sequence_name, frames_list, 'threemdot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):             # 必须按顺序放，因为代码里给排序了
        sequence_list = [
                        "md3001-1",
                        "md3002-1",
                        "md3003-1",
                        "md3004-1",
                        "md3006-1",
                        "md3007-1",
                        "md3009-1",
                        "md3010-1",
                        "md3011-1",
                        "md3012-1",
                        "md3014-1",
                        "md3015-1",
                        "md3021-1",
                        "md3022-1",
                        "md3023-1",
                        "md3024-1",
                        "md3025-1",
                        "md3028-1",
                        "md3029-1",
                        "md3033-1",
                        "md3037-1",
                        "md3039-1",
                        "md3041-1",
                        "md3042-1",
                        "md3043-1",
                        "md3045-1",
                        "md3046-1",
                        "md3047-1",
                        "md3049-1",
                        "md3052-1",
                        "md3053-1",
                        "md3056-1",
                        "md3057-1",
                        "md3061-1",
                        "md3063-1",
                        "md3001-2",
                        "md3002-2",
                        "md3003-2",
                        "md3004-2",
                        "md3006-2",
                        "md3007-2",
                        "md3009-2",
                        "md3010-2",
                        "md3011-2",
                        "md3012-2",
                        "md3014-2",
                        "md3015-2",
                        "md3021-2",
                        "md3022-2",
                        "md3023-2",
                        "md3024-2",
                        "md3025-2",
                        "md3028-2",
                        "md3029-2",
                        "md3033-2",
                        "md3037-2",
                        "md3039-2",
                        "md3041-2",
                        "md3042-2",
                        "md3043-2",
                        "md3045-2",
                        "md3046-2",
                        "md3047-2",
                        "md3049-2",
                        "md3052-2",
                        "md3053-2",
                        "md3056-2",
                        "md3057-2",
                        "md3061-2",
                        "md3063-2",
                        "md3001-3",
                        "md3002-3",
                        "md3003-3",
                        "md3004-3",
                        "md3006-3",
                        "md3007-3",
                        "md3009-3",
                        "md3010-3",
                        "md3011-3",
                        "md3012-3",
                        "md3014-3",
                        "md3015-3",
                        "md3021-3",
                        "md3022-3",
                        "md3023-3",
                        "md3024-3",
                        "md3025-3",
                        "md3028-3",
                        "md3029-3",
                        "md3033-3",
                        "md3037-3",
                        "md3039-3",
                        "md3041-3",
                        "md3042-3",
                        "md3043-3",
                        "md3045-3",
                        "md3046-3",
                        "md3047-3",
                        "md3049-3",
                        "md3052-3",
                        "md3053-3",
                        "md3056-3",
                        "md3057-3",
                        "md3061-3",
                        "md3063-3",
                        ]
        return sequence_list
