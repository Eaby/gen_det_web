import torch
import torch.nn as nn
import torch.optim as optim
import timm
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import os
from tqdm import tqdm
from PIL import UnidentifiedImageError
from PIL import ImageFile
from pytorch_lightning.callbacks import ModelCheckpoint

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        
        # This implementation pre-filters the dataset to only include valid images,
        # which is more efficient than checking in __getitem__.
        self.samples = self._filter_images(self.samples)
        self.targets = [s[1] for s in self.samples]

    def _filter_images(self, samples):
        valid_samples = []
        print(f"Validating images in {self.root}...")
        for image_path, target in tqdm(samples):
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Verify that it is a valid image
                valid_samples.append((image_path, target))
            except (OSError, UnidentifiedImageError, FileNotFoundError):
                print(f"Warning: Skipping corrupt or missing image: {image_path}")
                continue
        return valid_samples

# Define the LightningModule for the Custom ResNet50 model
class CustomResNet50(pl.LightningModule):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        # Create the model with pretrained=False to avoid network calls here.
        # Your gradio_app will load the actual weights from your local .ckpt file.
        self.base_model = timm.create_model('resnet50', pretrained=False)
        self.base_model.reset_classifier(0)
        
        # Global average pooling and custom classification head for BINARY classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1) # Single output for binary classification
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')

    def forward(self, x):
        x = self.base_model.forward_features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs).squeeze(-1)
        loss = self.criterion(outputs, labels.float())
        acc = self.train_accuracy(torch.sigmoid(outputs), labels.int())
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs).squeeze(-1)
        loss = self.criterion(outputs, labels.float())
        acc = self.val_accuracy(torch.sigmoid(outputs), labels.int())
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Define the LightningDataModule to handle the dataset and dataloaders
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, batch_size=32, num_workers=4):
        super(ImageNetDataModule, self).__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        self.train_dataset = CustomImageFolder(root=self.train_dir, transform=self.train_transforms)
        self.val_dataset = CustomImageFolder(root=self.val_dir, transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

if __name__ == "__main__":
    # Set up the training and validation directories
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    val_dir = '../data/gen_image/midjourney/Midjourney-20241023T111515Z-015/Midjourney/imagenet_midjourney/val'
    train_dir = '../data/gen_image/midjourney/Midjourney-20241023T111515Z-015/Midjourney/imagenet_midjourney/train'

    # Use a batch size that is a power of 2 for efficiency
    data_module = ImageNetDataModule(train_dir, val_dir, batch_size=64, num_workers=os.cpu_count())

    # Instantiate the model (no 'num_classes' argument needed)
    model = CustomResNet50()

    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.strategies import DDPStrategy

    logger = TensorBoardLogger('tb_logs', name='my_model_mid')
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
        every_n_epochs=10,
        dirpath='checkpoints/',
        filename='mid_resnet50-{epoch:02d}',
        save_weights_only=False,
    )

    # To resume, set the path here. e.g., 'checkpoints/mid_resnet50-epoch=19.ckpt'
    checkpoint_path = None

    # Use the 'nccl' backend for faster GPU communication
    trainer = pl.Trainer(
        max_epochs=31, 
        devices=2, 
        accelerator='gpu', 
        strategy=DDPStrategy(find_unused_parameters=False, process_group_backend='nccl'), 
        logger=logger,
        callbacks=[checkpoint_callback]
    )
    
    trainer.fit(model, data_module, ckpt_path=checkpoint_path)