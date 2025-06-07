import numpy as np
import scipy.stats as st
from scipy.stats import shapiro


def print_metrics(accuracy, losses, f1_scores, precision_scores, recall_scores, fold_metrics):
    print(200 * ' ')
    print(25 * '*', 'Average metrics', 25 * '*')
    print('Average Test Loss: {:.4f}, Average Test Acc: {:.4f}'.format(np.mean(losses), np.mean(accuracy)))
    print('Average F1 score:', np.mean(f1_scores))
    print('Average precision score:', np.mean(precision_scores))
    print('Average recall score:', np.mean(recall_scores))
    print('Accuracy variance :', np.var(accuracy))
    print('F1 variance :', np.var(f1_scores))
    print('Max Test acc: {:.4f}, Min Test Acc: {:.4f}'.format(np.max(accuracy), np.min(accuracy)))

    print(200 * ' ')
    print(25 * '*', 'Fold table', 25 * '*')
    print(f"{'Metric':<10}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1':<10}{'Loss':<10}")
    for row in fold_metrics:
        print(f"{row[0]:<10}{row[1]:<10.3f}{row[2]:<10.3f}{row[3]:<10.3f}{row[4]:<10.3f}{row[5]:<10.3f}")


def confidence_interval(metric_values, confidence=0.95):
    metric_values = np.array(metric_values)
    n = len(metric_values)
    mean = np.mean(metric_values)
    std_err = np.std(metric_values, ddof=1) / np.sqrt(n)
    interval = st.t.interval(confidence, df=n-1, loc=mean, scale=std_err)
    return mean, interval


def print_confidence_interval(metric_name, metric):
    mean, ci = confidence_interval(metric)
    print(f"\n{metric_name}: {mean:.2f}% (95% CI: {ci[0]:.2f}% – {ci[1]:.2f}%)")


def print_shapiro_value(accuracy, metric_name):
    stat, p = shapiro(accuracy)
    print(f"\nShapiro-Wilk p-value of {metric_name}: {p:.4f} →",
          "✅ Distribution is likely normal." if p > 0.05 else "❌ Distribution is not normal.")