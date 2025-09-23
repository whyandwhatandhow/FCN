import torch
import torch.nn as nn
import copy
import time
import os
import matplotlib.pyplot as plt
from model_and_data import get_voc_dataloader_local, FCN_ResNet, NUM_CLASSES


def model_train_process(model, train_loader, num_epochs, save_path="best_model.pth", device="cpu"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # VOC背景有时是255
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    train_loss_all = []
    train_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_pixels = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # [N, num_classes, H, W]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # loss 累加
            running_loss += loss.item() * images.size(0)

            # accuracy 计算
            preds = torch.argmax(outputs, dim=1)  # [N, H, W]
            correct = (preds == targets).float().sum()
            running_correct += correct.item()
            total_pixels += targets.numel()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_correct / total_pixels

        train_loss_all.append(epoch_loss)
        train_acc_all.append(epoch_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

        # 保存最好模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best Train Acc: {best_acc:.4f}")

    # 加载最好权重
    model.load_state_dict(best_model_wts)

    # 保存为 .pth
    torch.save(model.state_dict(), save_path)
    print(f"Best model saved at {os.path.abspath(save_path)}")

    return model, train_loss_all, train_acc_all


def visualize_training(train_loss_all, train_acc_all):
    epochs = range(1, len(train_loss_all) + 1)

    plt.figure(figsize=(12,5))

    # Loss 曲线
    plt.subplot(1,2,1)
    plt.plot(epochs, train_loss_all, 'b-', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    # Accuracy 曲线
    plt.subplot(1,2,2)
    plt.plot(epochs, train_acc_all, 'g-', label="Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Curve")
    plt.legend()

    plt.show()



if __name__ == "__main__":
    voc_dir = "./VOC2012"
    train_loader = get_voc_dataloader_local(voc_dir, split="train", batch_size=4)

    model = FCN_ResNet(num_classes=NUM_CLASSES)

    trained_model, train_loss_all, train_acc_all = model_train_process(
        model, train_loader, num_epochs=20, save_path="fcn_resnet_voc.pth"
    )

    visualize_training(train_loss_all, train_acc_all)

