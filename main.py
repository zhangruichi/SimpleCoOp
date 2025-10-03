import torchvision.transforms as transforms
import torch
import argparse
import os
import random
import numpy as np

from datasets import StanfordCars
from CoOp import CoOp


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    seed_everything(args.SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    
    resolution = 224
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = StanfordCars(root=args.data_root, num_shots=args.num_shots, seed=args.data_seed, split="train", transform=transform_train)
    test_dataset = StanfordCars(root=args.data_root, split="test", transform=transform_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CoOp(train_dataset.classes, model_name=args.model_name, n_ctx=args.n_ctx, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    

    for epoch in range(args.epochs):
        model.train()
        model.base_model.eval()
        for item_idx, (imgs, labels) in enumerate(train_dataloader):
            imgs, labels = imgs.to("cuda:0"), labels.to("cuda:0")
            logits = model(imgs)
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch},idx {item_idx}, loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
        scheduler.step()

        if epoch % 10 == 0:
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for item_idx, (imgs, labels) in enumerate(test_dataloader):
                    preds = model(imgs.to("cuda:0")).argmax(dim=-1).cpu()
                    all_preds.append(preds)
                    all_labels.append(labels)
                    print(f"Eval Epoch {epoch}, idx {item_idx}/{len(test_dataloader)}")
                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                acc = (all_preds == all_labels).float().mean()
                print(f"Epoch {epoch}, Test Acc: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training settings
    parser.add_argument("--gpu", type=int, default=3, help="GPU id to use.")
    parser.add_argument("--SEED", type=int, default=0, help="random seed")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="num workers")

    # datasets Settings
    parser.add_argument("--data_root", type=str, default="Path/To/stanford_cars", help="dataset root to stanford_cars")
    parser.add_argument("--num_shots", type=int, default=16, help="number of shots")
    parser.add_argument("--data_seed", type=int, default=0, help="random seed for data")

    # model settings
    parser.add_argument("--model_name", type=str, default="RN50", help="model type")
    parser.add_argument("--n_ctx", type=int, default=16, help="number of context tokens")

    args = parser.parse_args()
    main(args)
