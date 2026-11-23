import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CattleTripletDataset(Dataset):

    def __init__(self, root_dir, split='train'):

        self.root_dir = root_dir

        self.transform_train = transforms.Compose([

            transforms.Resize((128, 128)),

            transforms.RandomResizedCrop(
                105,
                scale=(0.75, 1.0)
            ),

            transforms.RandomHorizontalFlip(p=0.5),

            transforms.RandomRotation(10),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.transform_val = transforms.Compose([

            transforms.Resize((105, 105)),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.transform = (
            self.transform_train
            if split == 'train'
            else self.transform_val
        )

        # confusion bank from error analysis
        self.confusion_bank = {

            "11": ["14"],
            "14": ["11"],

            "16": ["13"],
            "13": ["16"],

            "23": ["10"],
            "10": ["23"],

            "6": ["3"],
            "3": ["6"],

            "9": ["6"],

            "7": ["10", "23"],

            "20": ["19"]
        }

        identities = sorted(os.listdir(root_dir))

        split_index = int(
            0.8 * len(identities)
        )

        if split == 'train':
            identities = identities[:split_index]
        else:
            identities = identities[split_index:]

        self.id_to_images = {}
        self.image_paths = []

        for identity in identities:

            folder = os.path.join(
                root_dir,
                identity
            )

            if not os.path.isdir(folder):
                continue

            images = [

                os.path.join(folder, img)

                for img in os.listdir(folder)
            ]

            if len(images) < 2:
                continue

            self.id_to_images[
                identity
            ] = images

            for img in images:

                self.image_paths.append(
                    (img, identity)
                )

        self.all_ids = list(
            self.id_to_images.keys()
        )

        print(
            f"{split.upper()} SET: "
            f"{len(self.image_paths)} images"
        )

    def __len__(self):

        return len(
            self.image_paths
        ) * 5

    def __getitem__(self, index):

        anchor_path, anchor_label = random.choice(
            self.image_paths
        )

        positive_candidates = [

            img for img in
            self.id_to_images[
                anchor_label
            ]

            if img != anchor_path
        ]

        positive_path = random.choice(
            positive_candidates
        )

        # 30% hard negative
        use_hard = random.random() < 0.3

        if use_hard and anchor_label in self.confusion_bank:

            hard_choices = [

                x for x in
                self.confusion_bank[
                    anchor_label
                ]

                if x in self.id_to_images
            ]

            if hard_choices:

                negative_label = random.choice(
                    hard_choices
                )

            else:

                negative_label = random.choice([

                    x for x in
                    self.all_ids

                    if x != anchor_label
                ])

        else:

            negative_label = random.choice([

                x for x in
                self.all_ids

                if x != anchor_label
            ])

        negative_path = random.choice(
            self.id_to_images[
                negative_label
            ]
        )

        anchor = Image.open(
            anchor_path
        ).convert("RGB")

        positive = Image.open(
            positive_path
        ).convert("RGB")

        negative = Image.open(
            negative_path
        ).convert("RGB")

        anchor = self.transform(
            anchor
        )

        positive = self.transform(
            positive
        )

        negative = self.transform(
            negative
        )

        return anchor, positive, negative