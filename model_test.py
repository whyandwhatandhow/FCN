import torch
from torch.utils.data import DataLoader
from model_and_data import VOC2012SegLocal  # 你写的 VOC Dataset 类
from model_and_data import FCN_ResNet        # 你写的模型
import numpy as np
from PIL import Image
import os

NUM_CLASSES = 21  # VOC2012 全类别（含背景）

# --------- 计算 IoU 和 Pixel Acc ---------
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_pixels = 0
    conf_matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)  # [N, 21, H, W]
            preds = outputs.argmax(dim=1)

            # Pixel Accuracy
            total_correct += (preds == targets).sum().item()
            total_pixels += targets.numel()

            # Confusion matrix
            for t, p in zip(targets.view(-1), preds.view(-1)):
                if t < NUM_CLASSES:
                    conf_matrix[t.long(), p.long()] += 1

    pixel_acc = total_correct / total_pixels

    iou_list = []
    for i in range(NUM_CLASSES):
        tp = conf_matrix[i, i].item()
        fn = conf_matrix[i, :].sum().item() - tp
        fp = conf_matrix[:, i].sum().item() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0
        iou_list.append(iou)

    mean_iou = sum(iou_list) / NUM_CLASSES
    return pixel_acc, mean_iou, iou_list


# --------- 可视化预测 ---------
def save_predictions(model, dataloader, device, save_dir="predictions"):
    os.makedirs(save_dir, exist_ok=True)
    colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()  # [N, H, W]

            for i, pred in enumerate(preds):
                h, w = pred.shape
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(NUM_CLASSES):
                    color_mask[pred == c] = colors[c]
                Image.fromarray(color_mask).save(os.path.join(save_dir, f"pred_{idx*len(preds)+i}.png"))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    model = FCN_ResNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load("checkpoints/best.pth", map_location=device))
    model.to(device)

    # 加载数据
    val_loader = DataLoader(
        VOC2012SegLocal(voc2012_dir="VOCdevkit/VOC2012", split="val", resize_size=512),
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    # 评估
    pixel_acc, mean_iou, iou_list = evaluate(model, val_loader, device)
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    for i, iou in enumerate(iou_list):
        print(f"Class {i} IoU: {iou:.4f}")

    # 保存预测可视化
    save_predictions(model, val_loader, device, save_dir="predictions")
    print("预测结果已保存到 predictions/ 文件夹")
