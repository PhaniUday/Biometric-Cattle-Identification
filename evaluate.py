import torch
import numpy as np
from torch.utils.data import DataLoader
from model import CattleMuzzleNet
from data_loader import CattleSiameseDataset
from sklearn.metrics import roc_curve, f1_score
import matplotlib.pyplot as plt

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CattleMuzzleNet().to(device)
    model.load_state_dict(torch.load("cattle_muzzle_net_final.pth"))
    model.eval()

    test_dataset = CattleSiameseDataset(split='test')
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    distances = []
    labels = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2 = img1.to(device), img2.to(device)
            dist = model(img1, img2)
            distances.extend(dist.cpu().numpy())
            labels.extend(label.numpy())

    distances = np.array(distances)
    labels = np.array(labels)

    # Calculate ROC Curve and EER (Equal Error Rate)
    # Note: In our model, a smaller distance means "Same Cow" (Label 1)
    # So we use (1 - normalized_distances) for the ROC calculation
    fpr, tpr, thresholds = roc_curve(labels, 1 - (distances / np.max(distances)))
    
    # Finding EER: where False Positive Rate approx equals False Negative Rate
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fpr - fnr)))]
    eer = fpr[np.nanargmin(np.absolute((fpr - fnr)))]

    print(f"\n--- FINAL METRICS ---")
    print(f"EER (Equal Error Rate): {eer*100:.2f}%")
    
    # Plotting the results for your documentation
    plt.plot(fpr, tpr, label=f'ROC Curve (EER: {eer*100:.2f}%)')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('CattleMuzzleNet Performance')
    plt.legend()
    plt.savefig('evaluation_results.png')
    print("Results saved to evaluation_results.png")

if __name__ == "__main__":
    evaluate()