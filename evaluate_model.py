import os
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

register_coco_instances("hands_test", {}, "path/to/labeled_test_annotations.json", "/path/to/test_images")

dataset_dicts_valid = DatasetCatalog.get("hands_test")
hands_metadata_valid = MetadataCatalog.get("hands_test")

cfg = get_cfg()
cfg.merge_from_file("configs/test_evaluator.yaml")
cfg.DATASETS.TEST = ("hands_test",)

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

trainer = MyTrainer(cfg)
trainer.resume_or_load(True)
evaluator = COCOEvaluator("hands_test", cfg, False, output_dir=cfg.OUTPUT_DIR, use_fast_impl=False)
test_loader = build_detection_test_loader(cfg, "hands_test")

results = inference_on_dataset(trainer.model, test_loader, evaluator)

print("All results:", results, "\n")
print("AP50:", results['bbox']["AP50"])
print("AP75:", results['bbox']["AP75"])