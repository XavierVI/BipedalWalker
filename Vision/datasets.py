import os
import torch
from torch.utils.data import Dataset
import torchvision
from pycocotools.coco import COCO
from PIL import Image


class CocoDoomDataset(torchvision.datasets.CocoDetection):
    """
    Custom COCO dataset for training a DETR model.
    """

    def __init__(self, data_dir, annotation_file_name, processor):
        """
        Args:
            data_dir: Path to dataset
            processor: DETR image processor
        """
        # load COCO annotation
        annotation_file = os.path.join(
            data_dir, annotation_file_name)
        super().__init__(root=data_dir, annFile=annotation_file)

        self.img_folder = data_dir
        self.coco = COCO(annotation_file)
        self.processor = processor
        self.id2label = self.coco.loadCats(self.coco.getCatIds())
        self.ids = list(sorted(self.coco.imgs.keys()))

        print(f"Loaded {annotation_file_name}")
        print(f"Number of images: {len(self.coco.imgs)}")
        print(f"Number of Categories: {len(self.coco.getCatIds())}")

    def __len__(self):
        return len(self.ids)

    def get_preprocessed_item(self, idx):
        """
        Get a single preprocessed image and target.
        
        Uses preprocessed data from disk.
        """
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(
            self.img_folder, "preprocessed", img_info['file_name'])
        img_path = img_path.replace('.png', '.pt')
        
        print(f"Loading preprocessed data from: {img_path}")
        
        data = torch.load(img_path, weights_only=False)
        pixel_values = data['pixel_values']
        target = data['labels']
        
        return pixel_values, target

    def __getitem__(self, idx):
        """
        Get a single image and its target.

        Uses the processor.
        """
        img, target = super(CocoDoomDataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        target = {'image_id': img_id, 'annotations': target}

        # preprocess data
        encoding = self.processor(
            images=img,
            annotations=target,
            return_tensors="pt"
        )
        
        # squeeze and [0] to remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target


    def get_image(self, idx):
        """
        Get a single image by index.
        """
        # For DETR models, we can use this line of code to
        # obtain appropriate structure for the annotations
        img, target = super(CocoDoomDataset, self).__getitem__(idx)
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        target = {'image_id': img_id, 'annotations': target}
        return img, target, img_info["file_name"]
