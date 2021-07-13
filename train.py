import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_loader
from utils import initialize_model, plot_barchart
import test

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            losses = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    labels = labels.view(-1, 1)
                    loss = criterion(outputs, labels.float())

                    preds = torch.round(torch.sigmoid(outputs))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                losses.append(loss.item())
                preds = preds.view(-1, 1)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) *100

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

### 改num_epochs, 是否預訓練, figure名稱, model名稱, 資料集比例

### Parameters ###
# attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Eyeglasses']
# attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Orange_Hair', 'Yellow_Hair']
attrs = ['Eyeglasses']
dataset = 'cartoon'
model_list = ['AlexNet']
num_classes = 1
batch_size = 16
num_epochs = 10
input_size = 224

feature_extract = True
pretrained = True

if dataset == 'celeba':
    crop_size = 178
else:
    crop_size = 400

path = f'../datasets/{dataset}/images'
attr_path = f'../datasets/{dataset}/list_attr_{dataset}.txt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup the loss fxn
criterion = nn.BCEWithLogitsLoss()

figure_data = {'Attributes':[], 'Model':[], 'Attribute Discriminator Accuracy':[]}

if __name__ == '__main__':
    for attr in attrs:
        for model_name in model_list:
            # Initialize model
            model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=pretrained)
            # model = StyleEncoder(image_size=input_size, n_layers=2)
            # print(model)

            # Send the model to GPU
            model = model.to(device)

            params_to_update = model.parameters()
            print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name, param in model.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\t",name)
            else:
                for name, param in model.named_parameters():
                    if param.requires_grad == True:
                        print("\t",name)

            # Observe that all parameters are being optimized
            optimizer_ft = optim.Adam(params_to_update, lr=0.0002, betas=(0.5, 0.999))

            ### Initializing Datasets and Dataloaders 
            dataloaders_dict = get_loader(path, attr_path, [attr], crop_size, input_size)

            # Train and evaluate
            model, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model=="inception"))
            # Test data show cufusion matrix
            # load_model(model, '../output_pth/{}_cartoon-{}(F).pkl'.format(attr, model_name))
            test(model, dataloaders_dict['test'])
            # Save model
            model_path = '../output_pth/{}_{}-{}.pkl'.format(attr, dataset, model_name)
            torch.save(model.state_dict(), model_path)
            print("Save {} model!!!".format(model_path))

            ## Plot Figure
            figure_data['Attributes'].append(attr.replace('_', ' '))
            figure_data['Model'].append(model_name)
            figure_data['Attribute Discriminator Accuracy'].append(max(hist).item())

    # Plot figure
    plot_barchart(f'../MDGAN/experiments/Acc_figures/non-pre-trained_{dataset}.png', figure_data)