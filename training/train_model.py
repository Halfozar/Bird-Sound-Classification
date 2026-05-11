import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from app.utils.feature_extraction import extract_features

# ============================================
# DATASET PATH
# ============================================

DATASET_PATH = "../dataset"

X = []
y = []

classes = os.listdir(DATASET_PATH)

# ============================================
# FEATURE EXTRACTION
# ============================================

for label in classes:

    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    print(f"Processing {label}")

    for file_name in os.listdir(folder_path):

        if file_name.endswith(".mp3") or file_name.endswith(".wav"):

            file_path = os.path.join(folder_path, file_name)

            features = extract_features(file_path)

            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# ============================================
# TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ============================================
# SCALING
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================
# MODEL
# ============================================

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

model.fit(X_train, y_train)

# ============================================
# EVALUATION
# ============================================

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# ============================================
# SAVE MODEL
# ============================================

os.makedirs("../app/model", exist_ok=True)

joblib.dump(model, "../app/model/bird_svm_model.pkl")
joblib.dump(scaler, "../app/model/scaler.pkl")

print("Model Saved Successfully")