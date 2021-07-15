
import csv
import glob
import pandas as pd
from shutil import copyfile
from torch.utils import data
from torchvision import transforms as T, datasets

## 依照原csv分類cartoonset
def classifyByCsv(path='./datasets/cartoon/cartoonset10k/'):
    others = [0, 1, 108, 109, 110] #不要的屬性圖片

    for csvfile_path in glob.glob(path + '*.csv'):
        df = pd.read_csv(csvfile_path, header=None, index_col=0)
        glasses = df.loc['glasses'][1]
        color = df.loc['hair_color'][1]
        style = df.loc['hair'][1]
        img_path = csvfile_path.replace('csv', 'png')
        filename = img_path[33:]
    #     new_name = str(i).zfill(6) + '.png'

    #     ## Glasses ##
    #     if not (color == 8 or color == 9 or style in others):
    #         if glasses == 11: ## 11是沒戴眼鏡
    #             new_path = './datasets/cartoon_folder/imagefolder/Eyeglasses_filter/0/{}'.format(filename)
    #         else:
    #             new_path = './datasets/cartoon_folder/imagefolder/Eyeglasses_filter/1/{}'.format(filename)
    #         copyfile(img_path, new_path)
        
        ## Hair color ##
        if style in others:
            new_path = './datasets/cartoon_folder/cartoonset/others/{}'.format(filename)
        else:
            new_path = './datasets/cartoon_folder/cartoonset/{}/{}'.format(color, filename)
        copyfile(img_path, new_path)
        
    print("Done")
        
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

# 利用imageFolder寫attr.csv檔案
def create_list_attr():

    transform = T.Compose([
        T.ToTensor()
    ])
    dataset_ = datasets.ImageFolder('./datasets/cartoon_folder/', transform)
    dataloader = data.DataLoader(dataset_, batch_size=1, shuffle=False)

    print(dataset_.class_to_idx) #{'Black_Hair': 0, 'Blond_Hair': 1, 'Brown_Hair': 2, 'Grey_Hair': 3, 'Orange_Hair': 4, 'Others': 5, 'White_Hair': 6, 'Yellow_Hair': 7}
    with open('list_attr_cartoon.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['FileName', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Eyeglasses', 'others'])
        ## Positive dataset
        for i, (img, c) in enumerate(dataset_.imgs):
            fname, _ = dataloader.dataset.samples[i]
            fname = fname[-10:]
            n = [0] * 12
            n[c] = 1
            n.insert(0, fname)
            writer.writerow([fname, c])
                
    print("Done!!!")

if __name__ == '__main__':
    classifyByCsv()