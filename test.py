from train import Cartoon, DataLoader, T, classification_report, initialize_model, load_model
import torch
from tqdm import tqdm

def get_loader(image_dir, attr_path, selected_attrs, crop_size=378, image_size=224, 
               batch_size=16):
    """Build and return a data loader."""

    dataloader = {}
    transform = []
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))

    test_transform = T.Compose(transform)
    test_dataset = Cartoon(image_dir, attr_path, selected_attrs, test_transform, mode='test')
    dataloader['test'] = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return dataloader

def set_crop_size(dataset):
    if dataset == 'celeba':
        crop_size = 178
    else:
        crop_size = 400
    return crop_size

def test(model, test_loader):
    model = model.to(device)

    pred_list = []
    true_list = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            pred = torch.round(torch.sigmoid(pred))

            pred_list.append(pred.cpu().int().numpy())
            true_list.append(labels.cpu().int().numpy())

    pred_list = [a.squeeze().tolist() for a in pred_list]
    true_list = [b.squeeze().tolist() for b in true_list]

    print(classification_report(true_list, pred_list, labels=[0, 1]))

if __name__ == '__main__':
    dataset = 'celeba'
    crop_size = set_crop_size(dataset)
    attr = 'Brown_Hair'
    model_name = 'AlexNet'
    num_classes = 1
    
    feature_extract = False
    pretrained = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=pretrained)
    dataloaders_dict = get_loader(f'../datasets/{dataset}/images', f'../datasets/{dataset}/list_attr_{dataset}.txt', [attr], crop_size, 224)
    load_model(model, '../output_pth/{}_{}-{}(F).pkl'.format(attr, dataset, model_name))
    model = model.to(device)

    model.eval()
    pred_list = []
    true_list = []
    with torch.no_grad():
        for x, c in dataloaders_dict['test']:
            x = x.cuda()
            c = c.cuda()

            pred = model(x)
            pred = torch.round(torch.sigmoid(pred))

            pred_list.append(pred.cpu().int().numpy())
            true_list.append(c.cpu().int().numpy())

        pred_list = [a.squeeze().tolist() for a in pred_list]
        true_list = [b.squeeze().tolist() for b in true_list]

    print(classification_report(true_list, pred_list, labels=[0, 1]))
