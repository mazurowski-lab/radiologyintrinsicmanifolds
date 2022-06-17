import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image

# constants
label_csvs = {
    'chexpert' :'data/chexpert/CheXpert-v1.0/train_subset.csv',
    'mura' :'data/mura/MURA-v1.1/train_image_paths.csv',
    'rsna' : 'data/rsna/stage_2_train.csv'
}

data_dirs = {
    'chexpert' : 'data/chexpert/CheXpert-v1.0/subset/train',
    'dbc' : 'data/dbc/png_subset',
    'oai' : 'data/oai/OAI_img',
    'brats' : 'data/brats/Brats_normalized/flair',
    'mura' : 'data/mura/',
    'rsna' : 'data/rsna/stage_2_train_png',
    'prostate' : 'data/prostate/train_png'
}

class MedicalDataset(Dataset):
    def __init__(self, label_csv, data_dir, img_size, transform):
        self.label_csv = label_csv
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        
        # to be initialized by child class
        self.labels = None
                 
    def normalize(self, img):
        # normalize to range [0, 255]
        # img expected to be array
                 
        # uint16 -> float
        img = img.astype(np.float) * 255. / img.max()
        # float -> unit8
        img = img.astype(np.uint8)
        
        return img
    
    def __getitem__(self, idx):
        
        fpath, target  = self.labels[idx]
        
        # print(fpath)
        
        # load img from file (png or jpg)
        img_arr = io.imread(fpath, as_gray=True)
        
        # print(img_arr.shape)
        
        # normalize
        img_arr = self.normalize(img_arr)
        
        # print(img_arr)
        
        # convert to tensor
        data = torch.from_numpy(img_arr)
        # print(data, type(data))
        data = data.type(torch.FloatTensor) 
       
        # add channel dim
        data = torch.unsqueeze(data, 0)
        
        # resize to standard dimensionality
        data = transforms.Resize((self.img_size, self.img_size))(data)
        # bilinear by default
        
        # do any data augmentation/training transformations
        if self.transform:
            data = self.transform(data)
        
        return data, target
    
    def __len__(self):
        return len(self.labels)
    
    def get_avg_extrinsic_dim(self):
        EDs = []
        for label in tqdm(self.labels):
            fpath, _  = label
            img_arr = io.imread(fpath, as_gray=True)
            EDs.append(img_arr.size)
            
        return np.mean(EDs)
    
    def resize_imgs_on_disk(self, targetsize=224):
        print('resizing images to {}x{}...'.format(targetsize,targetsize))
        for label in tqdm(self.labels):
            fpath, _  = label
            
            im = Image.open(fpath)
            im = im.resize((targetsize, targetsize), Image.BILINEAR)
            im.save(fpath)
            # print(fpath)
            # print('saved {}.'.format(fpath))

# individual datasets                   
class CheXpertDataset(MedicalDataset):
    def __init__(self, img_size, labeling='Pleural Effusion', train_transform=None):
        super(CheXpertDataset, self).__init__(label_csvs['chexpert'], data_dirs['chexpert'], img_size, train_transform)
        
        label_df = pd.read_csv(self.label_csv)
        labels = [] 
        # (fname, value = label (0 = neg, 1 = pos) )
        
        pos_ct = 0
        neg_ct = 0
        
        print('building CheXpert dataset.')
        for row_idx, row in label_df.iterrows():
            fname = row['New Path']
            fpath = os.path.join(self.data_dir, fname)
            
            target = row[labeling]
            if np.isnan(target):
                target = 0
                neg_ct += 1
            elif int(target) == -1:
                target = 0 
                neg_ct += 1
            else:
                target = 1
                pos_ct += 1
            assert target in [0, 1]
            
            labels.append((fpath, target))
            
        self.labels = labels
        print('{} positives, {} negatives.'.format(pos_ct, neg_ct))
        
        
class DBCDataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None):
        super(DBCDataset, self).__init__(None, data_dirs['dbc'], img_size, train_transform)

        labels = []
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building DBC dataset.')
        if labeling == 'default':
            for target, target_label in enumerate(['neg', 'pos']):
                case_dir = os.path.join(self.data_dir, target_label)
                for fname in os.listdir(case_dir):
                    if '.png' in fname:
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, target))
        else:
            raise NotImplementedError
            
        self.labels = labels
         
            
class OAIDataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None):
        super(OAIDataset, self).__init__(None, data_dirs['oai'], img_size, train_transform)

        labels = []
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building OAI dataset.')
        if labeling == 'default':
            for target in range(2):
                case_dir = os.path.join(self.data_dir, str(target))
                for fname in os.listdir(case_dir):
                    if '.png' in fname:
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, target))
        else:
            raise NotImplementedError
            
        self.labels = labels
        
        
class BraTSDataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None):
        super(BraTSDataset, self).__init__(None, data_dirs['brats'], img_size, train_transform)

        labels = []
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building BraTS dataset.')
        if labeling == 'default':
            for target in range(2):
                case_dir = os.path.join(self.data_dir, str(target))
                for fname in os.listdir(case_dir):
                    if '.jpg' in fname:
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, target))
        else:
            raise NotImplementedError
            
        self.labels = labels
        
        
class MURADataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None):
        super(MURADataset, self).__init__(label_csvs['mura'], data_dirs['mura'], img_size, train_transform)
        
        label_df = pd.read_csv(self.label_csv, names=['path'])
        labels = [] 
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building MURA dataset')
        for row_idx, row in label_df.iterrows():
            fname = row['path']
            fpath = os.path.join(self.data_dir, fname)
            
            target = None
            if 'negative' in fname:
                target = 0
            elif 'positive' in fname:
                target = 1
                
            assert target in [0, 1]
            
            labels.append((fpath, target))
            
        self.labels = labels
        
        
# RSNA intracranial hemorrhage brain CT
class RSNADataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None):
        super(RSNADataset, self).__init__(label_csvs['rsna'], data_dirs['rsna'], img_size, train_transform)
        
        
        # default labeling  = 0 for no hemorrage, 1 for any hemorrage
        if labeling == 'default':
            labeling = 'any'
        label_df = pd.read_csv(self.label_csv)
        label_df = label_df[label_df['ID'].str.contains(labeling)] 
        # get rid of irrelevant labels
        
        # display(label_df)
        
        labels = [] 
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building RSNA dataset')
        for row_idx, row in label_df.iterrows():
            fname = row['ID'].split('_')[:2]
            fname = '_'.join(fname) + '.png'
            fpath = os.path.join(self.data_dir, fname)
            
            target = int(row['Label'])         
            assert target in [0, 1]
            
            labels.append((fpath, target))
            
        self.labels = labels
        # print(self.labels)
        

class ProstateMRIDataset(MedicalDataset):
    def __init__(self, img_size, labeling='default', train_transform=None):
        super(ProstateMRIDataset, self).__init__(None, data_dirs['prostate'], img_size, train_transform)
        
        labels = []
        # (fname, value = label (0 = neg, 1 = pos) )
        print('building ProstateMRI dataset.')
        for score in range(6):
            case_dir = os.path.join(self.data_dir, str(score))

            if labeling == 'default':
                if score < 2:
                    target = 0
                else:
                    target = 1
                    
                    
            elif labeling == 'hard':
                # negative is least severe cancer, positive is more severe
                if score == 2:
                    target = 0
                elif score > 2:
                    target = 1
                else:
                    continue
            else:
                raise NotImplementedError
                    
            for fname in os.listdir(case_dir):
                if '.png' in fname:
                    fpath = os.path.join(case_dir, fname)
                    labels.append((fpath, target))
            
        self.labels = labels




# utils
def get_datasets(dataset_name, labeling='default', subset_size=None, train_frac=None, test_size=None, img_size=224):
    # either (1) specify train_frac, which split of subset to create train and test sets, or
    # (2) specify test_size
    
    if labeling != 'default':
        print('using non-default {} labeling.'.format(labeling))

    # first, option of getting subset of full dataset stored
    # then, option of splitting what's left into train and test
    # create dataset
    if dataset_name == 'chexpert':
        # default labeling is by Pleural Effusion state
        if labeling == 'default':
            labeling = 'Pleural Effusion'
        dataset = CheXpertDataset(img_size, labeling)
    elif dataset_name == 'dbc':
        dataset = DBCDataset(img_size, labeling)
    elif dataset_name == 'oai':
        dataset = OAIDataset(img_size, labeling)
    elif dataset_name == 'brats':
        dataset = BraTSDataset(img_size, labeling)
    elif dataset_name == 'mura':
        dataset = MURADataset(img_size, labeling)
    elif dataset_name == 'rsna':
        dataset = RSNADataset(img_size, labeling)
    elif dataset_name == 'prostate':
        dataset = ProstateMRIDataset(img_size, labeling)
    else:
        raise NotImplementedError
        
    if subset_size:
        # each class should have subset_size//2 instances
        # so remove extras
        pos_ct = 0
        neg_ct = 0
        class_size = subset_size//2
        new_labels = []
        # print(len(dataset.labels))
        for idx, label in enumerate(dataset.labels):
            if label[1] == 1 and pos_ct < class_size:
                new_labels.append(label)
                pos_ct += 1
            elif label[1] == 0 and neg_ct < class_size:
                new_labels.append(label)
                neg_ct += 1
                
        # print(new_labels)
        # print(len(new_labels))
        assert len(new_labels) == subset_size
        dataset.labels = new_labels
        print('{} positive examples, {} negative examples.'.format(pos_ct, neg_ct))
            
    # split into train and test if chosen
    if train_frac or test_size:
        if train_frac:
            train_size = int(train_frac * len(dataset))
            test_size = len(dataset) - train_size
        else:
            train_size = len(dataset) - test_size 

        # print(train_size, test_size)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1337))
        
#         train_dataset = MedicalDataset(train_transform)
#         val_dataset = MedicalDataset(val_transform)
#         indices = list(range(len(dataset.labels)))
#         # print(indices)
#         train_indices, test_indices = train_test_split(indices, train_size=train_frac)
#         train_dataset = torch.utils.data.Subset(dataset, train_indices)
#         test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
#         train_dataset.transform = train_transform
        
        return train_dataset, test_dataset
    else:
        # dataset.train_transform = train_transform
        return dataset