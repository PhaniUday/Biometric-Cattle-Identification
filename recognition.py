import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

from model import CattleMuzzleNet


def get_transform():

    # MUST match validation/inference transform
    return transforms.Compose([

        transforms.Resize((105, 105)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_embedding(model, image_path, transform, device):

    img = Image.open(image_path).convert("RGB")

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():

        embedding = model(img)

    return embedding.cpu().numpy()[0]


def run_recognition():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nRunning on:",
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    DATA_PATH = os.path.join(
        "Cattle Muzzle - DB",
        "Original"
    )

    # ----------------------------
    # LOAD MODEL
    # ----------------------------

    print("\nLoading Triplet model...")

    model = CattleMuzzleNet().to(device)

    model.load_state_dict(
        torch.load("triplet_best_model.pth")
    )

    model.eval()

    transform = get_transform()

    # ----------------------------
    # BUILD GALLERY + QUERY
    # ----------------------------

    gallery_db = {}
    query_set = []

    identities = sorted(os.listdir(DATA_PATH))

    print("\nBuilding gallery/query split...")

    for identity in identities:

        folder = os.path.join(
            DATA_PATH,
            identity
        )

        if not os.path.isdir(folder):
            continue

        images = [

            os.path.join(folder, img)

            for img in os.listdir(folder)
        ]

        if len(images) < 3:
            continue

        random.shuffle(images)

        split_idx = int(
            0.7 * len(images)
        )

        gallery_images = images[:split_idx]
        query_images = images[split_idx:]

        # query images
        for q in query_images:

            query_set.append(
                (q, identity)
            )

        # gallery embeddings
        embeddings = []

        for img in gallery_images:

            emb = extract_embedding(
                model,
                img,
                transform,
                device
            )

            embeddings.append(emb)

        # average embedding
        mean_embedding = np.mean(
            embeddings,
            axis=0
        )

        gallery_db[identity] = mean_embedding

    print(
        f"\nGallery cattle: {len(gallery_db)}"
    )

    print(
        f"Query images: {len(query_set)}"
    )

    # ----------------------------
    # RECOGNITION BENCHMARK
    # ----------------------------

    correct = 0
    total = len(query_set)

    mistakes = []

    intra_distances = []
    inter_distances = []

    print("\nRunning recognition benchmark...")

    for query_path, true_label in tqdm(query_set):

        query_embedding = extract_embedding(
            model,
            query_path,
            transform,
            device
        )

        distances = {}

        for gallery_label in gallery_db:

            gallery_embedding = gallery_db[
                gallery_label
            ]

            dist = np.linalg.norm(
                query_embedding -
                gallery_embedding
            )

            distances[
                gallery_label
            ] = dist

            # distance analysis
            if gallery_label == true_label:
                intra_distances.append(dist)
            else:
                inter_distances.append(dist)

        predicted_label = min(
            distances,
            key=distances.get
        )

        min_distance = distances[
            predicted_label
        ]

        if predicted_label == true_label:

            correct += 1

        else:

            mistakes.append({

                "true": true_label,

                "predicted": predicted_label,

                "distance": min_distance
            })

    # ----------------------------
    # METRICS
    # ----------------------------

    rank1_accuracy = (
        correct / total
    ) * 100

    avg_intra = np.mean(
        intra_distances
    )

    avg_inter = np.mean(
        inter_distances
    )

    threshold = (
        avg_intra + avg_inter
    ) / 2

    # ----------------------------
    # RESULTS
    # ----------------------------

    print("\n========== RESULTS ==========")

    print(
        f"Rank-1 Accuracy: "
        f"{rank1_accuracy:.2f}%"
    )

    print(
        f"Average SAME cow distance: "
        f"{avg_intra:.4f}"
    )

    print(
        f"Average DIFFERENT cow distance: "
        f"{avg_inter:.4f}"
    )

    print(
        f"Suggested threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"Total mistakes: "
        f"{len(mistakes)}"
    )

    print("============================")

    # ----------------------------
    # WRONG PREDICTIONS
    # ----------------------------

    print("\nSAMPLE WRONG PREDICTIONS:\n")

    for m in mistakes[:20]:

        print(

            f"TRUE: {m['true']}"

            f" → PREDICTED: {m['predicted']}"

            f" | DISTANCE: {m['distance']:.4f}"
        )


if __name__ == "__main__":
    run_recognition()