import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'mdot_test'
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
# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_baseline'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_mdot'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_fin30_baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_fin30_baseline'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_fin30_mdot', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_fin30_mdot'))


# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot2s', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_mdot2s'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_fin30_mdot2s', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_fin30_mdot2s'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot_train_15', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_mdot_train_15'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot_train_30', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_mdot_train_30'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot_train_45', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_mdot_train_45'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot_train_30fin', dataset_name=dataset_name,
#                             run_ids=None, display_name='OSTrack384_mdot_train_30fin'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot_train_xin20', dataset_name=dataset_name,                  # 当前最高49.72， 59.15
#                             run_ids=None, display_name='OSTrack384_mdot_train_xin20'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot_train_xin30', dataset_name=dataset_name,            
#                             run_ids=None, display_name='OSTrack384_mdot_train_xin30'))

trackers.extend(trackerlist(name='ostrack', parameter_name='useful/vitb_384_mae_ce_32x4_ep300_20_49.00', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='OSTrack384_20_49.00'))

trackers.extend(trackerlist(name='ostrack', parameter_name='useful/vitb_384_mae_ce_32x4_ep300_mdot_train_best20_49.72_49.52', dataset_name=dataset_name,                  # baseline 49.00
                            run_ids=None, display_name='OSTrack384_20_49.52'))

trackers.extend(trackerlist(name='ostrack', parameter_name='multi_matching/vitb_384_mae_ce_32x4_ep300_mdot_train_best20_49.72_50.30', dataset_name=dataset_name,                  
                            run_ids=None, display_name='OSTrack384_mdot_train_best_49.72_50.30'))

trackers.extend(trackerlist(name='ostrack', parameter_name='multi_matching/multi/vitb_384_mae_ce_32x4_ep300_mdot_train_best20_49.72_50.37', dataset_name=dataset_name,                  
                            run_ids=None, display_name='OSTrack384_mdot_train_best_49.72_50.37'))

trackers.extend(trackerlist(name='ostrack', parameter_name='multi_matching/multi/vitb_384_mae_ce_32x4_ep300_mdot_train_best20_49.72_double_50.58', dataset_name=dataset_name,                  
                            run_ids=None, display_name='OSTrack384_mdot_train_best_49.72_double_50.58'))

trackers.extend(trackerlist(name='ostrack', parameter_name='multi_matching/double_center/vitb_384_mae_ce_32x4_ep300_mdot_train_best20_49.72_50.75', dataset_name=dataset_name,              # 50.75但是融合结果较低    
                            run_ids=None, display_name='OSTrack384_mdot_train_best_49.72_double_center_50.75'))

trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot_train_best20_49.72_50.84', dataset_name=dataset_name,                
                            run_ids=None, display_name='OSTrack384_mdot_train_best_49.72_50.84'))

trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300_mdot_train_best20_49.72_double', dataset_name=dataset_name,                  
                            run_ids=None, display_name='OSTrack384_mdot_train_best_49.72_double'))

dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
