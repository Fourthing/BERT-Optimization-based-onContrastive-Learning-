import matplotlib.pyplot as plt


def plot_metrics(training_metrics):
    """绘制训练过程中损失、对齐性、均匀性的变化曲线"""
    epochs = range(1, len(training_metrics["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失变化曲线
    plt.subplot(1, 3, 1)
    plt.plot(epochs, training_metrics["loss"], label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')

    # 绘制对齐性变化曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, training_metrics["alignment"], label='Alignment', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Alignment')
    plt.title('Alignment over Epochs')

    # 绘制均匀性变化曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, training_metrics["uniformity"], label='Uniformity', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Uniformity')
    plt.title('Uniformity over Epochs')

    plt.tight_layout()
    plt.show()
