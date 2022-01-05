#!/usr/bin/env python
# coding: utf-8

# # RSNA影像前處理

# In[2]:


import os
root = "C:/Users/User/OneDrive/桌面/NYCU/final_project"
os.chdir(root)


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import itertools
import mmcv
import json


# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from efficientnet_pytorch import EfficientNet


# In[5]:


import warnings
warnings.filterwarnings('ignore')


# In[6]:


torch.backends.cudnn.enabled


# In[7]:


print(torch.__version__)


# In[76]:


get_ipython().system('nvidia-smi')


# In[66]:


import GPUtil
GPUtil.showUtilization()


# In[8]:


#True,說明 GPU驅動和 CUDA可以支持 pytorch的加速計算！
torch.cuda.is_available()


# In[10]:


import argparse
parser = argparse.ArgumentParser()
# settings
# parser.add_argument("--dir_name", type=str)
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument("--model", type=str, default='efficientnet-b2')
parser.add_argument("--device_id", type=list, default=[0,1,2,3])
parser.add_argument("--folds", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--grad_clip", type=int, default=1)
parser.add_argument("--test", action='store_true', default=False)
# learning
# parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--epochs1", type=int, default=5)
parser.add_argument("--epochs2", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_lr1", type=float, default=1e-2)
parser.add_argument("--max_lr2", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-6)
args = parser.parse_args(args=[])


# In[11]:

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")


# In[12]:

df = pd.read_csv('stage_2_train.csv')


# # Read_data

# In[16]:


from PIL import Image
class DataSet_for_classification(Dataset):
    def __init__(self, dataframe, test = False, transform = None):
        """
        Args:
            image_list_file: path to the file containing images with corresponding labels.
            test: (bool) – If True, image_list_file is for test
        """
        self.df = dataframe.drop_duplicates(subset=['patientId'], keep='first')
        self.df.reset_index(drop=True, inplace=True)
#         self.df = dataframe
        self.test = test
        self.transform = transform

              
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        if self.test == True:
            ID = self.df.iloc[index, 0]
            image = Image.open('C:/Users/User/OneDrive/桌面/mmdetection-master/data/RSNA/test/' + ID + '.jpg').convert('RGB')
        else:
            ID, labels = self.df.iloc[index, [0,6]].values
            image = Image.open('C:/Users/User/OneDrive/桌面/mmdetection-master/data/RSNA/train/' + ID + '.jpg').convert('RGB')
#         image = Image.open('JPG_Global/' + c_eng + '/' + Filename.replace('.dcm', '.jpg')).convert('RGB')
#         plt.imshow(image)
        if self.transform is not None:
            image = self.transform(image)
#         trans_image.show()
        if self.test == True:
            return image, ID
        else:
            return image, torch.tensor(labels), ID
    
    def __len__(self):
        return len(self.df)


# In[17]:


tran = DataSet_for_classification(df)
tran[0]


# # EfficientNet

# In[21]:

class Efficientnet(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard ResNet50
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size, grad = False):
        super(Efficientnet, self).__init__()
        self.network = EfficientNet.from_pretrained(args.model, num_classes = out_size)

    def forward(self, x):
        x = self.network(x)
        return x
    
    def freeze(self):
        # To freeze the residual layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network._fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        # Unfreeze all layers
        for param in self.network.parameters():
            param.require_grad = True


# In[24]:


def draw_chart(chart_data, val = True):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # -------------- draw loss image --------------
    ax[0].plot(chart_data['epoch'],chart_data['train_loss'],label='train_loss')
    if val == True:
        ax[0].plot(chart_data['epoch'],chart_data['val_loss'],label='val_loss')
    ax[0].grid(True,axis="y",ls='--')
    ax[0].legend(loc= 'best')
    ax[0].set_title('Loss on Training and Validation Data', fontsize=10)
    ax[0].set_xlabel('epoch',fontsize=10)
    ax[0].set_ylabel('Loss',fontsize=10)

    # -------------- draw accuracy image --------------
    ax[1].plot(chart_data['epoch'],chart_data['train_acc'],label='train_acc')
    if val == True:
        ax[1].plot(chart_data['epoch'],chart_data['val_acc'],label='val_acc')
    ax[1].grid(True,axis="y",ls='--')
    ax[1].legend(loc= 'best')
    ax[1].set_title('Accuracy  on Training and Validation Data',fontsize=10)
    ax[1].set_xlabel('epoch',fontsize=10)
    ax[1].set_ylabel('Accuracy',fontsize=10)       
    
    plt.tight_layout()

#%%
def class_weights(df):
    df = df.drop_duplicates(subset=['patientId'], keep='first')
    df.reset_index(drop=True, inplace=True)
    class_num = [Counter(df['Target'])[i] for i in range(args.n_classes)]
    print('class_numbers [label 0, label1] : {}'.format(class_num))
    class_weight = len(df) / torch.FloatTensor(class_num)
    # print(class_weight)
    # class_weight = class_weight / class_weight.sum()
    print('class_weigth [label 0, label1] : {}'.format(class_weight))
    return class_weight

# ### Training

# In[32]:


def fit(model, dataloader, optimizer, criterion, scheduler, chart_data, 
        acc_list, pred_train, pred_val, path_save, freeze = True):
    
    if freeze == True:
        epochs = args.epochs1
    else:
        epochs = args.epochs2
    total_epochs = args.epochs1 + args.epochs2

    for epoch in trange(epochs, desc="Epochs"):
        if freeze == True:
            chart_data['epoch'].append(epoch)
            print(f'Starting epoch {epoch+1}')
        else:
            chart_data['epoch'].append(epoch+args.epochs1)
            print(f'Starting epoch {epoch+args.epochs1+1}')            
        for phase in ['train', 'val']:
            if phase == "train":    
                model.train()
            else:    
                model.eval()

            running_loss = 0.0
            running_accuracy = 0.0
            predict = []
            ID_list = []
            for i, (data , target, ID) in enumerate(dataloader[phase]):
                data, target = data.to(device), target.to(device)
                if phase == 'train':
                    optimizer.zero_grad()
                    
                output = model(data)
                _, preds = torch.max(output, dim=1)
                predict.extend(preds.cpu())
                ID_list.extend(ID)
                loss = criterion(output, target)
                
                if phase == 'train':
                    loss.backward()
                    # Gradient clipping
                    if args.grad_clip: 
                        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                    optimizer.step()                       
                    
                # statistics              
                running_loss += loss.item()
                running_accuracy += preds.eq(target).sum().item()
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            epoch_acc = running_accuracy / len(dataloader[phase].dataset) 
            
            # visulize loss and accuracy
            if phase == 'train':
                chart_data['train_loss'].append((epoch_loss))
                chart_data['train_acc'].append(epoch_acc)
                curr_lr = optimizer.param_groups[0]['lr']
                print(f'LR:{curr_lr}')
                scheduler.step()
                pred_train.append([predict, ID_list])
                
            if phase == 'val':
                chart_data['val_loss'].append((epoch_loss))
                chart_data['val_acc'].append(epoch_acc)
                acc_list.append(epoch_acc)
                pred_val.append([predict, ID_list])
                
            if freeze == True:
                print('========================================================================') 
                print('Epoch [%d/%d]:%s Loss of the model:  %.4f %%' % (epoch+1, total_epochs, phase, 100 * epoch_loss))
                print('Epoch [%d/%d]:%s Accuracy of the model: %.4f %%' % (epoch+1, total_epochs,phase, 100 * epoch_acc)) 
                print('========================================================================')  
            else:
                print('========================================================================') 
                print('Epoch [%d/%d]:%s Loss of the model:  %.4f %%' % (epoch+args.epochs1+1, total_epochs, phase, 100 * epoch_loss))
                print('Epoch [%d/%d]:%s Accuracy of the model: %.4f %%' % (epoch+args.epochs1+1, total_epochs,phase, 100 * epoch_acc)) 
                print('========================================================================')                
                    
        if freeze == True:
            torch.save({'model_state_dict':model.state_dict()}, 
                       os.path.join(path_save,str(epoch+1)+'_'+str(epoch_acc)+'.pth'))              
        if freeze == False:
#             acc_list.append(epoch_acc)
            torch.save({'model_state_dict':model.state_dict()}, 
                       os.path.join(path_save,str(epoch+args.epochs1+1)+'_'+str(epoch_acc)+'.pth'))                   
    return chart_data, acc_list, pred_train, pred_val


# In[33]:


from tqdm import trange
# def training1(model , data_loader , optimizer, criterion, scheduler):
def training1_cv(data_loader, class_weight):
    since = time.time()
    path_save = "EfficientNet_weights_b2"
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    chart_data={"train_loss":[],"val_loss":[], "val_acc":[],"train_acc":[],"epoch":[]}
     
    model = Efficientnet(args.n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight = class_weight).to(device)
#     criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    acc_list = []
    pred_train = []
    pred_val = []
    print('******************************model freeze******************************')
    model.freeze()
    scheduler = lr_scheduler.OneCycleLR(optimizer, args.max_lr1, epochs = args.epochs1, 
                                        steps_per_epoch=len(dataloader['train']))
    chart_data, acc_list, pred_train, pred_val = fit(model, dataloader, optimizer, criterion, 
                                                     scheduler, chart_data, acc_list, pred_train, 
                                                     pred_val, path_save)


    print('*****************************model unfreeze*****************************')
    model.unfreeze()
    scheduler = lr_scheduler.OneCycleLR(optimizer, args.max_lr2, epochs = args.epochs2, 
                                        steps_per_epoch=len(dataloader['train']))
    chart_data, acc_list, pred_train, pred_val = fit(model, dataloader, optimizer, criterion, 
                                                     scheduler, chart_data, acc_list, pred_train, 
                                                     pred_val, path_save, freeze = False)        
  
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc for epoch {}: {:.4f}%'.format(np.argmax(acc_list)+1, max(acc_list) * 100))
    best_idx = np.argmax(acc_list)
    
    return chart_data, np.argmax(acc_list)+1, max(acc_list), pred_train[best_idx], pred_val[best_idx]


# In[36]:


def testing1(data_loader, best_epoch, accuracy, num_classes):
    model = Efficientnet(args.n_classes).to(device)
    path = "EfficientNet_weights_b2/" + str(best_epoch)+'_'+str(accuracy)+'.pth'
#     path = f'EfficientNet_weights_b4/model-fold-{fold}.pth'
    resume_file = torch.load(path, map_location = "cuda:0")
    model.load_state_dict(resume_file['model_state_dict'], False)
    model.eval()
     
    predicted = []
    with torch.no_grad():
        for j, (image, ID) in enumerate(data_loader):
            image = image.to(device)
            batch_prediction = model(image).cpu().detach()
#             _, pred = torch.max(batch_prediction, dim=1)
            pred = torch.argmax(batch_prediction, dim=1)
            predicted.extend(pred)
 
    return predicted

#%%

def model2_df(df, predict, train = False, test = False, id_list = None):
    if test == True:
        df['pred'] = predict
        diseases =  df[df['pred'] == 1]
        diseases =  diseases.iloc[:,[0,-1]]
        diseases.reset_index(drop=True, inplace=True)
        print('predict diseases : {}, normal : {}'.format(len(diseases), len(df) - len(diseases)))
        return diseases
    else:
        df_duplicate = df.drop_duplicates(subset=['patientId'], keep='first')
        df_duplicate.reset_index(drop=True, inplace=True)
        if train == True:   # 因為train dataset是隨機亂數，要以相同ID對應正確的預測值
            df_duplicate['patientId'] = df_duplicate['patientId'].astype('category')
            df_duplicate['patientId'].cat.reorder_categories(id_list, inplace=True)
            df_duplicate.sort_values('patientId', inplace=True)
            df_duplicate.reset_index(drop=True, inplace=True)
        df_duplicate['pred'] = predict  
        diseases =  df_duplicate[df_duplicate['pred'] == 1]
        diseases =  diseases.iloc[:,[0,-1]]
        diseases.reset_index(drop=True, inplace=True)
        print('predict diseases : {}, normal : {}'.format(len(diseases.patientId.unique()), 
                                                          len(df_duplicate.patientId.unique()) - len(diseases.patientId.unique())))
        df2 = df.merge(diseases, how='inner', on= 'patientId')
        df2.reset_index(drop=True, inplace=True)    
        dropnor_df2 = df2[df2['Target'] != 0]
        nor_df2 = df2[df2['Target'] == 0]
        print('gt_disease and pred_diseases :{}'.format(len(dropnor_df2.patientId.unique())))
        print('gt_normal  and pred_diseases :{}'.format(len(nor_df2.patientId.unique())))
        multi = df2.where(df2['patientId'].duplicated(keep = False) == True).dropna()
        print('multi_label image have :{}'.format(len(multi.patientId.unique())))
        return dropnor_df2

#%%

def create_dataset_dicts(df):
    coco_output = {
        "images" : [],
        "categories" : [],
        "annotations" : []
        }
    categories = [{'id' : 1, 'name' : 'Pneumonia'}]
    coco_output['categories'] = categories
    
    annotation_id = 0
    for image_id, img_name in enumerate(df.patientId.unique()):
        image_df = df[df.patientId == img_name]
        image_dict = {
            "file_name" : img_name + '.jpg', # file_name
            "height" : 1024,
            "width" : 1024,
            "id" : image_id
            } 
        coco_output['images'].append(image_dict)
    
        for _, row in image_df.iterrows():
            xmin = int(row.x)
            ymin = int(row.y)
            xmax = int(row.x + row.width)
            ymax = int(row.y + row.height)
            if xmin == ymin == 0:
                continue
            
            area = row.width * row.height
          
            poly = [
                (xmin, ymin), (xmax, ymin), 
                (xmax, ymax), (xmin, ymax)
            ]
            poly = list(itertools.chain.from_iterable(poly))
            # if xmin == ymin == 0:
            #   category = 0
            # else:
            #   category = 1
            # mode = 0
            mask_dict = {
                "id" : annotation_id,
                "image_id" : image_id,  
                "category_id" : 1,
                "bbox" : [xmin, ymin, xmax, ymax],
                "area" : area,
                "iscrowd" : 0,  # suppose all instances are not crowd
                "segmentation" : [poly],
                }
            coco_output["annotations"].append(mask_dict)
            # faster_dict = {
            #     "id" : annotation_id,
            #     "image_id" : image_id,  
            #     "category_id" : 1,
            #     "bbox": [xmin, ymin, xmax, ymax],
            #     "area" : area,
            #     "iscrowd": 0
            # }
            # coco_output["annotations"].append(faster_dict)
            # annotation_id += 1
    return coco_output

def create_testset_dicts(df):
    coco_output = {
        "images" : [],
        "categories" : [],
        }
    categories = [{'id' : 1, 'name' : 'Pneumonia'}]
    coco_output['categories'] = categories
    
    for image_id, img_name in enumerate(df.patientId.unique()):
        image_dict = {
            "file_name" : img_name.replace('.dcm','.jpg'), # file_name
            "height" : 1024,
            "width" : 1024,
            "id" : image_id
            } 
        coco_output['images'].append(image_dict)
    return coco_output


# In[70]:


if __name__ == '__main__':   
    test_df = pd.read_csv("stage_2_sample_submission.csv")
    train_indx, valid_indx = train_test_split(df.patientId.unique(), train_size=.9, random_state=7)
    train_df = df[df.patientId.isin(train_indx)]
    valid_df = df[df.patientId.isin(valid_indx)]
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    # print(f'train images : {len(train_df.patientId.unique())}, valid images : {len(valid_df.patientId.unique())}')  
    # train images : 24015, valid images : 2669
    
    test_transform = transforms.Compose([transforms.Resize([256,256]),
                                         transforms.ToTensor(),
                                         # transforms.ToPILImage()
                                         ])   
    train_transform = transforms.Compose([transforms.Resize([256,256]),
                                          transforms.RandomRotation(5),
                                          transforms.ToTensor(),
#                                               transforms.Normalize([0.485, 0.456, 0.406],
#                                                                    [0.229, 0.224, 0.225])
                                          ])
    
    test_set  = DataSet_for_classification(test_df, test = True, transform = test_transform)
    valid_set = DataSet_for_classification(valid_df, transform = test_transform) 
    train_set = DataSet_for_classification(train_df, transform = train_transform) 
    print('='*70) 
    print('For model 1 train/valid/test')
    print("train_df:", len(train_df), " / valid_df:",len(valid_df), " / test_df:",len(test_df))
    print('='*70)
    print("train_images:", len(train_set), " / valid_images: " , len(valid_set),  " / test_images:",len(test_set))    
    print('='*70)   
    
    dataloader  = {"train"      : DataLoader(dataset = train_set, batch_size = args.batch_size,
                                             shuffle = True, num_workers = args.num_workers),
                    "val"       : DataLoader(dataset = valid_set, batch_size = args.batch_size,
                                             shuffle = False, num_workers = args.num_workers),
                    "test"      : DataLoader(dataset = test_set, batch_size = args.batch_size,
                                             shuffle = False, num_workers = args.num_workers)}
    
    class_weight1 = class_weights(train_df)
    chart_data, best_epoch, best_acc, pred_train, pred_val = training1_cv(dataloader, class_weight1)
    draw_chart(chart_data)   
    
    pred = testing1(dataloader['test'], best_epoch, best_acc, 2)
    torch.cuda.empty_cache()
    
    print('-'*40)    
    print('Test')
    test_df2  = model2_df(test_df, pred, test = True)
    print('-'*40)
    print('Valid')
    valid_df2 = model2_df(valid_df, pred_val[0])
    print('-'*40)
    print('Train')
    train_df2 = model2_df(train_df, pred_train[0], train = True, id_list = pred_train[1])
    print('='*70) 
    print('For model 2 train/valid/test')
    print("train_df2:", len(train_df2), " / valid_df2:",len(valid_df2), " / test_df2:",len(test_df2))
    print("train_df2_images:", len(train_df2.patientId.unique()), 
          " / valid_df2_images:",len(valid_df2.patientId.unique()), 
          " / test_df2_images:",len(test_df2.patientId.unique()))
    print('='*70)
    
    
    mask_output =  create_testset_dicts(test_df2)
    mmcv.dump(mask_output, 'faster_test.json') 
    
    mask_output =  create_dataset_dicts(train_df2)
    mmcv.dump(mask_output, 'faster_train.json')  
    
    mask_output =  create_dataset_dicts(valid_df2)
    mmcv.dump(mask_output, 'faster_valid.json') 


    
    
        
