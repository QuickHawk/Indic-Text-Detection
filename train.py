import os
import argparse

from tqdm.auto import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms

from models import TrDBNet
from loss import DBLoss
from datetime import datetime

from utils import get_dataset

SUPPORTED_MODEL_LIST = {'TrDBNet': TrDBNet}
AVAILABLE_DATASETS = {'TotalText', 'SynthText'}
BACKBONE_OPTIONS = {'swin', 'convnext'}
BACKBONE_VARIANTS = {'tiny', 'small', 'base', 'large'}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training Script on the models")

    parser.add_argument("--name", type=str, default="TrDBNet",
                        help="Name of the model to be trained")

    parser.add_argument("--pretrained", type=bool, default=False,
                        help="Use pretrained model for training")
    parser.add_argument("--backbone", type=str, default="swin",
                        help="Backbone to use for UperNet")
    parser.add_argument("--variant", type=str, default="base",
                        help="Variant of the backbone to use")

    parser.add_argument("--model", type=str,
                        default="TrDBNet", help="Model to be trained")
    parser.add_argument("--dataset", type=str,
                        default="TotalText", help="Dataset to use to train")

    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-4, help="Learning rate for training")

    parser.add_argument("--gpu", type=bool, default=False,
                        help="Use GPU for training")

    parser.add_argument("--load_model", type=str, default=None,
                        help="Load the model from the path")
    # parser.add_argument("--save_model", type=str, default=None,
    #                     help="Save the model to the path")

    parser.add_argument("--save_interval", type=bool, default=True,
                        help="Save the model if the loss is minimum. Stored at './model_weights/checkpoints/ ")

    args = parser.parse_args()

    if args.model not in SUPPORTED_MODEL_LIST:
        raise ValueError("Model not found in the list of models. Please choose from {}".format(
            SUPPORTED_MODEL_LIST.keys()))

    if args.dataset not in AVAILABLE_DATASETS:
        raise ValueError("Dataset not found in the list of datasets. Please choose from {}".format(
            AVAILABLE_DATASETS))

    if args.backbone not in BACKBONE_OPTIONS:
        raise ValueError(
            "Backbone not found in the list of backbones. Please choose from {}".format(BACKBONE_OPTIONS))

    if args.variant not in BACKBONE_VARIANTS:
        raise ValueError(
            "Variant not found in the list of variants. Please choose from {}".format(BACKBONE_VARIANTS))

    # if os.path.exists(args.save_model):
    #     raise FileExistsError(f"The specified model path {
    #                           args.save_model} already exists.")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.learning_rate

    print("Loading Model : {}".format(args.model))
    model = SUPPORTED_MODEL_LIST[args.model](
        pretrained=args.pretrained, backbone=args.backbone, variant=args.variant)

    print("Loading Dataloader : {}".format(args.dataset))
    dataset = get_dataset(args.dataset, train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    if args.gpu == True:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            raise ValueError("GPU is not available. Please run on CPU")
    else:
        device = "cpu"

    print("Using Device : {}".format(device))
    model.to(device)
    
    if args.load_model is not None:
        if not os.path.exists(args.load_model):
            raise FileNotFoundError(f"The specified model path {
                                    args.load_model} does not exist.")

        model.load_state_dict(torch.load(args.load_model))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    criterion = DBLoss()

    model.train()

    loss_history = []

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        
        total_loss = 0

        progress_bar = tqdm(dataloader, desc="Training", dynamic_ncols=True, leave=True)
        progress_bar.set_description(f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (images, masks) in enumerate(progress_bar):

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            prob_map, thresh_map, bin_map = outputs

            loss = criterion(prob_map, thresh_map, bin_map, masks, masks)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                {"Batch": f"{batch_idx+1}/{len(dataloader)}", "Loss": loss.item()})

        avg_loss = total_loss/len(dataloader)
        tqdm.write(f"Epoch {epoch+1}/{EPOCHS} : Average Loss : {avg_loss}")

        if args.save_interval:
            if avg_loss < best_loss:
                best_loss = avg_loss

                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_name = f"./model_weights/checkpoints/{args.name}_{
                    args.backbone}_{args.variant}_epoch_{epoch+1}_{current_datetime}.pth"
                torch.save(model.state_dict(), file_name)

        loss_history.append(avg_loss)
    
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./model_weights/{args.name}_{
            args.backbone}_{args.variant}_{current_datetime}.pth"
    print("Saving Model at : {}".format(file_name))
    torch.save(model.state_dict(), file_name)
    
    print("Training Completed")
    print("Loss History : ", loss_history)