import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from src.plot import PlotUtils
from src.model.gexpnet import Gexpnet
from src.utils import convert_to_tensor, set_seed
from src.data.dataset import Database, load_mendeley_data
from src.metrics import print_confidence_interval, print_shapiro_value, print_metrics


SEED = 32
set_seed(SEED)

classes, x, y = load_mendeley_data()

le = LabelEncoder()
y_encoded = le.fit_transform(y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

k_folds = 10
kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)

accuracy, losses = [], []
precision_scores, recall_scores, f1_scores = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(x, y_encoded)):

    X_train, X_val = x.iloc[train_idx, :], x.iloc[val_idx, :]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', QuantileTransformer(output_distribution='normal', random_state=42)),
        ('feature_selection', SelectFromModel(
            estimator=LogisticRegression(
                penalty='l2',
                solver='newton-cg',
                C=0.1,
                class_weight='balanced',
                random_state=42
            ),
            threshold='median'
        )),
    ])

    X_train = pipeline.fit_transform(X_train, y_train)
    X_val = pipeline.transform(X_val)

    X_train_t, y_train_t = convert_to_tensor(X_train, y_train, device)
    X_val_t, y_val_t = convert_to_tensor(X_val, y_val, device)

    train_loader = DataLoader(Database(X_train_t, y_train_t), batch_size=64, shuffle=True)
    val_loader = DataLoader(Database(X_val_t, y_val_t), batch_size=len(y_val_t), shuffle=False)

    model = Gexpnet(X_train_t.shape[1], len(classes)).to(device)
    model.train_model(train_loader=train_loader, device=device)

    y_test, y_predicted_test, tst_acc, tst_loss = model.test_model(val_loader, device)

    f1 = f1_score(y_test, y_predicted_test, average='macro')
    precision = precision_score(y_test, y_predicted_test, average='macro')
    recall = recall_score(y_test, y_predicted_test, average='macro')

    print('Fold [{}/{}], Test Loss: {:.4f}, Test Acc: {:.4f}'.format(fold+1, k_folds, tst_loss, tst_acc))

    print(f"Per-class metrics for Fold {fold + 1}:\n")
    report = classification_report(y_test, y_predicted_test, target_names=list(classes), digits=3)
    print(report)

    # PlotUtils.plot_confusion_matrix(y_test, y_predicted_test, classes, fold, "Mendeley")

    accuracy.append(tst_acc)
    f1_scores.append(f1*100)
    precision_scores.append(precision*100)
    recall_scores.append(recall*100)
    losses.append(tst_loss)


fold_metrics = []
for i in range(k_folds):
    fold_metrics.append([f"Fold {i+1}", accuracy[i], precision_scores[i], recall_scores[i], f1_scores[i], losses[i]])

fold_metrics.append(["Mean ± SD", np.mean(accuracy), np.mean(precision_scores), np.mean(recall_scores),
                     np.mean(f1_scores), np.mean(losses)])
fold_metrics.append(["", np.std(accuracy), np.std(precision_scores), np.std(recall_scores), np.std(f1_scores),
                     np.std(losses)])

print_metrics(accuracy, losses, f1_scores, precision_scores, recall_scores, fold_metrics)

print_confidence_interval("Accuracy", accuracy)
print_confidence_interval("F1 Score", f1_scores)

print_shapiro_value(accuracy, "Accuracy")
print_shapiro_value(f1_scores, "F1 Score")

metrics_data = {
    "Accuracy": accuracy,
    "F1-score": f1_scores,
    "Precision": precision_scores,
    "Recall": recall_scores
}

df_metrics = pd.DataFrame(metrics_data)

# PlotUtils.plot_fold_metrics_boxplot(df_metrics, "Mendeley")
