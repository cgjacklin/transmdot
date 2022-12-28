from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/jack/OSTrack-main/data/got10k_lmdb'
    settings.got10k_path = '/jack/OSTrack-main/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/jack/OSTrack-main/data/itb'
    settings.lasot_extension_subset_path_path = '/jack/OSTrack-main/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/jack/OSTrack-main/data/lasot_lmdb'

    settings.lasot_path = '/jack/MDOT/Two-MDOT/train/two/'
    settings.mdot_test_path = '/jack/MDOT/Two-MDOT/test/two/'
    settings.threemdot_test_path = '/jack/MDOT/Three-MDOT/test/three/'
    
    settings.network_path = '/jack/OSTrack-main/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/jack/OSTrack-main/data/nfs'
    settings.otb_path = '/jack/OSTrack-main/data/otb'
    settings.prj_dir = '/jack/OSTrack-main'
    settings.result_plot_path = '/jack/OSTrack-main/output/test/result_plots'
    settings.results_path = '/jack/OSTrack-main/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/jack/OSTrack-main/output'
    settings.segmentation_path = '/jack/OSTrack-main/output/test/segmentation_results'
    settings.tc128_path = '/jack/OSTrack-main/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/jack/OSTrack-main/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/jack/OSTrack-main/data/trackingnet'
    settings.uav_path = '/jack/OSTrack-main/data/uav'
    settings.vot18_path = '/jack/OSTrack-main/data/vot2018'
    settings.vot22_path = '/jack/OSTrack-main/data/vot2022'
    settings.vot_path = '/jack/OSTrack-main/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

