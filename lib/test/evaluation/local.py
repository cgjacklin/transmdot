from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/root/workdir/transmdot/data/got10k_lmdb'
    settings.got10k_path = '/root/workdir/transmdot/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/root/workdir/transmdot/data/itb'
    settings.lasot_extension_subset_path_path = '/root/workdir/transmdot/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/root/workdir/transmdot/data/lasot_lmdb'

    settings.lasot_path = '/root/nas-public-linkdata/29150/LaSOT/LaSOTTest/LaSOTTest'
    settings.mdot_test_path = '/root/nas-public-linkdata/29150/MDOT/Two-MDOT/test/two/'
    settings.threemdot_test_path = '/root/nas-public-linkdata/29150/MDOT/Three-MDOT/test/three/'
    
    settings.network_path = '/root/workdir/transmdot/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/root/workdir/transmdot/data/nfs'
    settings.otb_path = '/root/workdir/transmdot/data/otb'
    settings.prj_dir = '//root/workdir/transmdot/'
    settings.result_plot_path = '/root/workdir/transmdot/output/test/result_plots'
    settings.results_path = '/root/workdir/transmdot/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/root/workdir/transmdot/output'
    settings.segmentation_path = '/root/workdir/transmdot/output/test/segmentation_results'
    settings.tc128_path = '/root/workdir/transmdot/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/root/workdir/transmdot/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/root/workdir/transmdot/data/trackingnet'
    settings.uav_path = '/root/workdir/transmdot/data/uav'
    settings.vot18_path = '/root/workdir/transmdot/data/vot2018'
    settings.vot22_path = '/root/workdir/transmdot/data/vot2022'
    settings.vot_path = '/root/workdir/transmdot/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

