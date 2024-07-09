import matplotlib.pyplot as plt

def training_curve(class1_acc, class2_acc, class3_acc, avg_ls):
     epochs = range(1, 26)
     fig, axs = plt.subplots(2, 2, figsize=(12, 10))

     # 绘制第一个子图：Class 1 Accuracy
     axs[0, 0].plot(epochs, class1_acc, label='Class 1 Accuracy', color='r')
     axs[0, 0].set_title('Class 1 Accuracy')
     axs[0, 0].set_xlabel('Epoch')
     axs[0, 0].set_ylabel('Accuracy')
     axs[0, 0].legend()

     # 绘制第二个子图：Class 2 Accuracy
     axs[0, 1].plot(epochs, class2_acc, label='Class 2 Accuracy', color='g')
     axs[0, 1].set_title('Class 2 Accuracy')
     axs[0, 1].set_xlabel('Epoch')
     axs[0, 1].set_ylabel('Accuracy')
     axs[0, 1].legend()

     # 绘制第三个子图：Class 3 Accuracy
     axs[1, 0].plot(epochs, class3_acc, label='Class 3 Accuracy', color='b')
     axs[1, 0].set_title('Class 3 Accuracy')
     axs[1, 0].set_xlabel('Epoch')
     axs[1, 0].set_ylabel('Accuracy')
     axs[1, 0].legend()

     # 绘制第四个子图：Average Loss
     axs[1, 1].plot(epochs, avg_ls, label='Average Loss', color='m')
     axs[1, 1].set_title('Average Loss')
     axs[1, 1].set_xlabel('Epoch')
     axs[1, 1].set_ylabel('Loss')
     axs[1, 1].legend()

     # 调整子图之间的间距
     plt.tight_layout()

     # 显示图形
     plt.show()
