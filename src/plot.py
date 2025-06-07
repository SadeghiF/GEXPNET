import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'Times New Roman'


class PlotUtils:

    @staticmethod
    def plot_fold_metrics_boxplot(df_metrics, dataset_name, save=None):
        df_melted = df_metrics.melt(var_name='Metric', value_name='Score')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Metric', y='Score', data=df_melted)
        plt.title(f"Boxplot of {dataset_name} across Folds")
        plt.ylabel("Score (%)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        if save:
            plt.savefig(f"{dataset_name}_Boxplot_of_Metrics_across_Folds.pdf", bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, fold, dataset_name, save=None):
        y_true_converted = []
        y_pred_converted = []
        for y_t, y_p in zip(y_true, y_pred):
            try:
                y_true_converted.append(str(classes[int(y_t)]))
                y_pred_converted.append(str(classes[int(y_p)]))
            except (ValueError, IndexError):
                print(f"Warning: Invalid index {y_t} or {y_p} for classes {classes}")
                y_true_converted.append("Unknown")
                y_pred_converted.append("Unknown")

        disp = ConfusionMatrixDisplay.from_predictions(
            y_true_converted,
            y_pred_converted,
            labels=classes,
            normalize='true',
            colorbar=False,
            cmap=plt.cm.Blues
        )

        cm = confusion_matrix(y_true_converted, y_pred_converted, labels=classes)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized, 0)
        for i in range(len(classes)):
            for j in range(len(classes)):
                count = cm[i, j]
                percentage = cm_normalized[i, j]
                text = f"{int(count)}\n({percentage:.2f})" if count > 0 else "0\n(0.00)"
                disp.text_[i, j].set_text(text)

        if save:
            plt.savefig(f"{dataset_name}_Confusion_Matrix_Fold_{fold + 1}.pdf", bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_per_class_bar_chart(y_true, y_pred, classes, fold, dataset_name, save=None):
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(len(classes)))
        x = np.arange(len(classes))
        width = 0.25

        plt.bar(x - width, precision, width=width, label='Precision')
        plt.bar(x, recall, width=width, label='Recall')
        plt.bar(x + width, f1, width=width, label='F1-score')

        plt.xticks(ticks=x, labels=classes)
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if save:
            plt.savefig(f"{dataset_name}_Class_Metrics_Fold_{fold + 1}.pdf", bbox_inches='tight')
        plt.show()
