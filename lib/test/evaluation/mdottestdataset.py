import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LaSOTDataset(BaseDataset):
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
        self.base_path = self.env_settings.mdot_test_path
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
        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list = [
                        'md2001-1',
                        'md2002-1',
                        'md2003-1',
                        'md2004-1',
                        'md2006-1',
                        'md2008-1',
                        'md2009-1',
                        'md2012-1',
                        'md2013-1',
                        'md2015-1',
                        'md2017-1',
                        'md2019-1',
                        'md2020-1',
                        'md2021-1',
                        'md2022-1',
                        'md2024-1',
                        'md2026-1',
                        'md2027-1',
                        'md2028-1',
                        'md2029-1',
                        'md2032-1',
                        'md2033-1',
                        'md2034-1',
                        'md2039-1',
                        'md2042-1',
                        'md2048-1',
                        'md2049-1',
                        'md2051-1',
                        'md2052-1',
                        'md2053-1',
                        'md2056-1',
                        'md2058-1',
                        'md2059-1',
                        'md2060-1',
                        'md2063-1',
                        'md2065-1',
                        'md2066-1',
                        'md2067-1',
                        'md2070-1',
                        'md2071-1',
                        'md2074-1',
                        'md2075-1',
                        'md2076-1',
                        'md2078-1',
                        'md2079-1',
                        'md2080-1',
                        'md2081-1',
                        'md2082-1',
                        'md2083-1',
                        'md2085-1',
                        'md2086-1',
                        'md2087-1',
                        'md2090-1',
                        'md2091-1',
                        'md2092-1',
                        'md2001-2',
                        'md2002-2',
                        'md2003-2',
                        'md2004-2',
                        'md2006-2',
                        'md2008-2',
                        'md2009-2',
                        'md2012-2',
                        'md2013-2',
                        'md2015-2',
                        'md2017-2',
                        'md2019-2',
                        'md2020-2',
                        'md2021-2',
                        'md2022-2',
                        'md2024-2',
                        'md2026-2',
                        'md2027-2',
                        'md2028-2',
                        'md2029-2',
                        'md2032-2',
                        'md2033-2',
                        'md2034-2',
                        'md2039-2',
                        'md2042-2',
                        'md2048-2',
                        'md2049-2',
                        'md2051-2',
                        'md2052-2',
                        'md2053-2',
                        'md2056-2',
                        'md2058-2',
                        'md2059-2',
                        'md2060-2',
                        'md2063-2',
                        'md2065-2',
                        'md2066-2',
                        'md2067-2',
                        'md2070-2',
                        'md2071-2',
                        'md2074-2',
                        'md2075-2',
                        'md2076-2',
                        'md2078-2',
                        'md2079-2',
                        'md2080-2',
                        'md2081-2',
                        'md2082-2',
                        'md2083-2',
                        'md2085-2',
                        'md2086-2',
                        'md2087-2',
                        'md2090-2',
                        'md2091-2',
                        'md2092-2',
                        ]
        return sequence_list
