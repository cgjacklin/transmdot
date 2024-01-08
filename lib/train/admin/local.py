class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/workdir/transmdot/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/root/workdir/transmdot/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/root/workdir/transmdot/pretrained_networks'

        # self.lasot_dir = '/home/visiondata/chenguanlin/LaSOT/LaSOT/LaSOTBenchmark/'
        self.lasot_dir = '/root/nas-public-linkdata/29150/LaSOT/LaSOT/LaSOTBenchmark/'
        self.twomdot_dir = '/root/nas-public-linkdata/29150/MDOT/Two-MDOT/train/two/'
        self.threemdot_dir = '/root/nas-public-linkdata/29150/MDOT/Three-MDOT/train/three/'

        self.got10k_dir = '/root/workdir/transmdot/data/got10k/train'
        self.got10k_val_dir = '/root/workdir/transmdot/data/got10k/val'
        self.lasot_lmdb_dir = '/root/workdir/transmdot/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/root/workdir/transmdot/data/got10k_lmdb'
        self.trackingnet_dir = '/root/workdir/transmdot/data/trackingnet'
        self.trackingnet_lmdb_dir = '/root/workdir/transmdot/data/trackingnet_lmdb'
        self.coco_dir = '/root/workdir/transmdot/data/coco'
        self.coco_lmdb_dir = '/root/workdir/transmdot/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/root/workdir/transmdot/data/vid'
        self.imagenet_lmdb_dir = '/root/workdir/transmdot/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
