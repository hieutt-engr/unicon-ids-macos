import numpy as np
import torch
import os
from torchvision import transforms
from PIL import Image
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import Dataset

class CANDataset(Dataset):
    def __init__(self, root_dir, window_size, is_train=True, include_data=False, transform=None):
        if is_train:
            self.root_dir = os.path.join(root_dir, 'train')
        else:
            self.root_dir = os.path.join(root_dir, 'val')
            
        self.include_data = include_data
        self.is_train = is_train
        self.transform = transform  # This will be TwoCropTransform
        self.window_size = window_size
        self.total_size = len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        # Load data as before
        filenames = '{}/{}.tfrec'.format(self.root_dir, idx)
        index_path = None
        # description = {'id_seq': 'int', 'data_seq': 'int', 'timestamp': 'int', 'label': 'int'}
        description = {'id_seq': 'int', 'data_seq': 'int', 'label': 'int'}
        dataset = TFRecordDataset(filenames, index_path, description)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        data = next(iter(dataloader))
        
        # id_seq, data_seq, timestamp, label = data['id_seq'], data['data_seq'], data['timestamp'], data['label']
        id_seq, data_seq, label = data['id_seq'], data['data_seq'], data['label']

        id_seq = id_seq.to(torch.float)
        data_seq = data_seq.to(torch.float)
        # timestamp = timestamp.to(torch.float)

        id_seq[id_seq == 0] = -1
        if id_seq.numel() == 1024 and data_seq.numel() == 1024:
            id_seq = id_seq.view(32, 32)
            data_seq = data_seq.view(32, 32)
            # timestamp = timestamp.view(32, 32)
        else:
            raise RuntimeError(f"Invalid tensor size for id_seq or data_seq")

        # Create 32x32x3 tensor by combining id_seq and data_seq
        combined_tensor = torch.stack([id_seq, data_seq], dim=-1)

        # Change direction from 32 x 32 x 2 to 2 x 32 x 32 [C, H, W]
        combined_tensor = combined_tensor.permute(2, 0, 1)

        # Make a fake channel 2 x 32 x 32 => 3 x 32 x 32
        dummy_channel = torch.zeros(1, 32, 32)
        combined_tensor = torch.cat((combined_tensor, dummy_channel), dim=0)
        

        # Apply transformations if provided
        if self.transform:
            combined_tensor = self.transform(combined_tensor)
        return combined_tensor, label[0][0]
        
    def __len__(self):
        return self.total_size
    
    