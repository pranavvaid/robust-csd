import os
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_consistency_train_loader
from detectron2.data import DatasetMapper, ConsistencyDatasetMapper
from detectron2.data import transforms as T
from detectron2.engine import DefaultConsistencyTrainer
from detectron2.evaluation import COCOEvaluator

'''SET UP DATASET'''

from detectron2.data.datasets import register_coco_instances

register_coco_instances("hands_train", {}, "path/to/labeled_train_annotations.json", "/path/to/train_images/")
register_coco_instances("hands_valid", {}, "path/to/labeled_validation_annotations.json", "/path/to/validation_images/")
# Unlabeled images ids are provided in COCO format, but associated annotations are not needed
register_coco_instances("consistency_train", {}, "path/to/unlabeled_images_ids.json", "/path/to/unlabeled_images/")

dataset_dicts_train = DatasetCatalog.get("hands_train")
hands_metadata_train = MetadataCatalog.get("hands_train")

dataset_dicts_valid = DatasetCatalog.get("hands_valid")
hands_metadata_valid = MetadataCatalog.get("hands_valid")

'''CONFIGURATION'''
cfg = get_cfg()

cfg.merge_from_file("configs/consistency_model_train_config.yaml")

cfg.DATASETS.TRAIN = ("hands_train",)
cfg.DATASETS.TEST = ("hands_valid",)  # no metrics implemented for this dataset

# Provide ids for unlabeled images
cfg.DATASETS.CONSISTENCY_TRAIN = ("consistency_train",)

# \tau, the confidence threshold for proposals
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
# k, the number of proposals retained
cfg.SOLVER.CONSISTENCY_TOPK = 5
# \epsilon, minimum IoU threshold for proposals to be retained
cfg.SOLVER.CONSISTENCY_IOU = 0.25
# lambda_u, the multiplicative weight on the consistency loss
cfg.SOLVER.CONSISTENCY_LOSS_WEIGHT = 10
# an optional parameter to control the iteration at which unlabeled data begins to be used
cfg.SOLVER.CONSISTENCY_LOSS_START_ITER = 0
# n, the number of unlabeled datapoints sampled per labeled datpoint
cfg.SOLVER.CONSISTENCY_DATA_RATIO = 8


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

'''TRAIN'''
class MyTrainer(DefaultConsistencyTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference_consistency")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[T.ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'), T.RandomFlip(), T.ConsistencyCutoutTransformHalf()])
        return build_detection_train_loader(cfg, mapper=mapper)
    @classmethod
    def build_consistency_train_loader(cls, cfg):
        mapper = ConsistencyDatasetMapper(cfg, is_train=True, img1_augmentations = [], img2_augmentations=[T.ConsistencyCutoutTransformAlways()])
        return build_detection_consistency_train_loader(cfg, mapper=mapper)

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()