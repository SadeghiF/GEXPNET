import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer


class Database(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def encoding_labels(y):
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    onehot_encoder = LabelBinarizer()
    onehot_encoded = onehot_encoder.fit_transform(label)
    return classes, onehot_encoded


def load_mendeley_data():
    data = pd.read_csv('src/data/mendeley/Data.csv', header=None)
    x = data.drop(data.columns[-1], axis=1)
    y = pd.read_csv('src/data/mendeley/Type.csv', header=None).squeeze()
    classes, y_long = encoding_labels(y)
    return classes, x, y
