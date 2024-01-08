import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'threemdot_test'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""


trackers.extend(trackerlist(name='ostrack', parameter_name='threemdot/vitb_384_mae_ce_32x4_ep300_threemdot', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='OSTrack384'))

trackers.extend(trackerlist(name='ostrack', parameter_name='threemdot/vitb_384_mae_ce_32x4_ep300_threemdot_best20', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='OSTrack384_best20_48.02'))

trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_threemdot', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='OSTrack384_threemdot_train_50.98'))

# trackers.extend(trackerlist(name='threemdot', parameter_name='three_mdot_train', dataset_name=dataset_name,                  # baseline 49.00
#                             run_ids=None, display_name='threemdot_train'))

trackers.extend(trackerlist(name='threemdot', parameter_name='three_mdot_train_2_50.79', dataset_name=dataset_name,                  # 40eph多模板+跨机重检测
                            run_ids=None, display_name='threemdot_train_2_50.79'))

trackers.extend(trackerlist(name='threemdot', parameter_name='three_mdot_train_2_49.2', dataset_name=dataset_name,                  # 40eph多模板，无跨机重检测
                            run_ids=None, display_name='threemdot_train_2_49.2'))

trackers.extend(trackerlist(name='threemdot', parameter_name='three_mdot_train_finbest10_48.24', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='threemdot_finbest10_48.24'))

trackers.extend(trackerlist(name='threemdot', parameter_name='three_mdot_train_finbest10_old', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='threemdot_finbest10'))

trackers.extend(trackerlist(name='threemdot', parameter_name='three_mdot_train_2_49.77', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='threemdot_train_2_baseline'))

trackers.extend(trackerlist(name='threemdot', parameter_name='three_mdot_train_2_51.35', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='threemdot_train_3_单模板有跨机'))

trackers.extend(trackerlist(name='threemdot', parameter_name='three_mdot_train_3', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='threemdot_train_3_多模板无跨机'))

trackers.extend(trackerlist(name='threemdot', parameter_name='three_mdot_train_3_51.79', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='threemdot_train_3_多模板有跨机'))



dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
