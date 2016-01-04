from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import pandas as pd
import numpy as np


if __name__ == '__main__':
    df = pd.read_json('data/images/compiled.json')
    X = df['img']
    X = list(X.map(lambda x: list(np.array(x).flatten())).values)
    X = np.array(X, dtype=np.uint8)
    y = df['label']

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X, y)

    joblib.dump(knn, 'models/knn.pkl')
