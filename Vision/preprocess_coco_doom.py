# Add project directory to path for imports
import torch
from transformers import DetrImageProcessor
from datasets import CocoDoomDataset
from PIL import Image
import sys
import os
sys.path.append(os.path.join(os.pardir))



# create preprocessor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# create dataset instance
train_dataset = CocoDoomDataset(
    data_dir=os.path.join(os.pardir, os.pardir, "datasets", "cocodoom"),
    annotation_file_name="run-train.json",
    processor=processor
)
val_dataset = CocoDoomDataset(
    data_dir=os.path.join(os.pardir, os.pardir, "datasets", "cocodoom"),
    annotation_file_name="run-val.json",
    processor=processor
)
test_dataset = CocoDoomDataset(
    data_dir=os.path.join(os.pardir, os.pardir, "datasets", "cocodoom"),
    annotation_file_name="run-test.json",
    processor=processor
)


def preprocess_and_save_dataset(dataset):
    for i in range(len(dataset)):
        # fetch the image
        image, target, img_file_name = dataset.get_image(i)
        # ann = dataset.get_annotation(i)

        print(f"Image {i} - file name: {img_file_name}")
        print(f"Target (before): {target}")

        # preprocess the image
        encoding = processor(
            images=image,
            annotations=target,
            return_tensors="pt"
        )

        pixel_values = encoding['pixel_values'].squeeze()
        target = dict(encoding['labels'][0])

        # reduce format of target tensors
        target['boxes'] = target['boxes'].to(torch.float16)
        target['size'] = None
        # we only have 94 categories
        target['class_labels'] = target['class_labels'].to(torch.int16)
        target['area'] = None  # remove area to save space
        target['iscrowd'] = None  # remove iscrowd to save space

        print(f"Pixel values shape: {pixel_values.shape}")
        print(f"Target: {target}")

        # modify file name to have .pt extension
        pt_file_name = os.path.splitext(img_file_name)[0] + ".pt"

        print(f"Image {i} processed with shape: {encoding['pixel_values'].shape}")
        save_path = os.path.join(
            os.pardir, os.pardir,
            "datasets", "cocodoom", "preprocessed", pt_file_name)

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping save.")
        else:
            print(f"Saving to: {save_path}")
            # save the processed data
            torch.save(
                {
                    "pixel_values": pixel_values,
                    "labels": target
                },
                save_path
            )


# Preprocess and save datasets
# print("Preprocessing and saving validation dataset...")
# preprocess_and_save_dataset(val_dataset)
print("Preprocessing and saving test dataset...")
preprocess_and_save_dataset(test_dataset)
# print("Preprocessing and saving training dataset...")
# preprocess_and_save_dataset(train_dataset)
