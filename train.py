"""
This file is used to train a vision transformer on the CocoDoom dataset.

CocoDoom: https://www.robots.ox.ac.uk/~vgg/research/researchdoom/cocodoom/
"""
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForObjectDetection, AdamW
from Vision.datasets import CocoDoomDataset
from Vision.Trainer import Trainer
import pycocotools
from pycocotools.coco import COCO


def collate_fn(batch):
    """Custom collate function to handle variable number of objects per image."""
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return torch.stack(pixel_values), labels

def create_dataloaders(processor):
    # Load COCO annotations
    coco_doom_dataset_path = os.path.join(
        os.pardir, os.pardir, "datasets", "cocodoom")
    
    # we're using the run split,
    # so run1 = train, run2 = val, and run3 = test
    train_annotation_file = "run-train.json"
    val_annotation_file = "run-val.json"
    test_annotation_file = "run-test.json"

    train_dataset = CocoDoomDataset(
        data_dir=coco_doom_dataset_path,
        annotation_file_name=train_annotation_file,
        processor=processor
    )
    val_dataset = CocoDoomDataset(
        data_dir=coco_doom_dataset_path,
        annotation_file_name=val_annotation_file,
        processor=processor
    )
    test_dataset = CocoDoomDataset(
        data_dir=coco_doom_dataset_path,
        annotation_file_name=test_annotation_file,
        processor=processor
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def main():
    optimizer_lr = 1e-5
    weight_decay = 1e-4
    batch_size = 4
    num_epochs = 10
    
    # Training setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    # Load processor
    processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50")

    # Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_dataloaders(processor)

    train_dataloader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = create_dataloader(
        val_dataset, batch_size=batch_size, shuffle=True)

    # Load the model
    model = AutoModelForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(train_dataset.coco.getCatIds()),
        id2label={i: cat['name'] for i, cat in enumerate(train_dataset.id2label)},
        label2id={cat['name']: i for i, cat in enumerate(train_dataset.id2label)}
    ).to(device)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_lr,
        weight_decay=weight_decay
    )
    # Create Trainer
    trainer = Trainer()
    trainer.train(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        num_epochs=num_epochs
    )

    print("\nTraining complete!")

if __name__ == "__main__":
    main()