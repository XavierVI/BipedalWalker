import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image


class CocoDoomDataset(Dataset):
    """Custom COCO dataset for DETR training."""

    def __init__(self, data_dir, annotation_file_name, processor):
        """
        Args:
            data_dir: Path to dataset
            processor: DETR image processor for preprocessing
        """
        # load COCO annotation
        annotation_file = os.path.join(
            data_dir, annotation_file_name)

        self.img_folder = data_dir
        self.coco = COCO(annotation_file)
        self.processor = processor
        self.id2label = self.coco.loadCats(self.coco.getCatIds())
        self.ids = list(sorted(self.coco.imgs.keys()))

        print(f"Loaded {annotation_file_name}")
        print(f"Training set: {len(self.coco.imgs)} images")
        print(f"Validation set: {len(self.coco.imgs)} images")
        print(f"Test set: {len(self.coco.imgs)} images")
        print(f"Number of Categories: {len(self.coco.getCatIds())}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Get image info
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get annotations in COCO format (xywh, absolute pixels)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Filter out crowd / invalid boxes and ensure required keys
        coco_annotations = []
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            if ann.get("bbox") is None:
                continue
            cat_contig = self.catid2contig[ann["category_id"]]
            coco_annotations.append({
                # [x, y, w, h] in absolute pixels
                "bbox": ann["bbox"],
                "category_id": cat_contig,       # remapped to contiguous ids
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0)
            })

        # Build target in the format expected by the processor
        target = {
            "image_id": img_id,
            "annotations": coco_annotations
        }

        # Process image and target with DETR processor
        encoding = self.processor(
            images=image, annotations=target, return_tensors="pt")

        # Remove batch dimension added by processor
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]

        return pixel_values, labels
