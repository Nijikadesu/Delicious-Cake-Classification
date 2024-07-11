## Delicious_Cake_Classification

这是我在XDUAI院开设课程《专业基础实践》中的选题<基于ResNet50实现多目标美味蛋糕图像分类>的项目代码与使用说明，是通过微调预训练ResNet50模型完成的针对蛋糕图像的多目标分类任务

This is a Multi-target classifiction task using pretrained ResNet50.

****
Environment required:
```angular2html
python 3.10
torch 2.2.2
torchvision 0.17.2
matplotlib 3.7.0
pandas 1.5.3
numpy 1.23.5
```
****
To fine-tune the ResNet50 model:
```angular2html
python main.py
```
To make inference:
```angular2html
python infer.py
```
****
Pretrained model are given in 'weights.pth'

Dataset is given in https://www.kaggle.com/datasets/rajkumarl/cakey-bakey?select=023.jpg, thanks for their contribution!
