from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import pandas as pd
import os
import sys

DATA_DIR = 'data'
OUTPUT_DIR = 'output'

required_files = ['train.csv', 'train_labels.csv', 'test.csv', 'test_users.csv', 'geo_info.csv', 'referer_vectors.csv']
missing_files = [f for f in required_files if not os.path.exists(os.path.join(DATA_DIR, f))]
if missing_files:
    print(f"Error: the following files were not found in directory {DATA_DIR}: {', '.join(missing_files)}", file=sys.stderr)
    sys.exit(1)

print("Loading data, working directory:", DATA_DIR)
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), sep=';')
labels = pd.read_csv(os.path.join(DATA_DIR, 'train_labels.csv'), sep=';')
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), sep=';')
test_users = pd.read_csv(os.path.join(DATA_DIR, 'test_users.csv'), sep=';')
geo = pd.read_csv(os.path.join(DATA_DIR, 'geo_info.csv'), sep=';').fillna({'region_id': 'none'})
ref_vec = pd.read_csv(os.path.join(DATA_DIR, 'referer_vectors.csv'), sep=';')

print("Data successfully loaded, shapes:")
print(f" - train: {train.shape}")
print(f" - labels: {labels.shape}")
print(f" - test: {test.shape}")
print(f" - test_users: {test_users.shape}")
print(f" - geo: {geo.shape}")
print(f" - referer_vectors: {ref_vec.shape}")

print("Processing features...")
ua_map = {ua: eval(ua) for ua in train['user_agent'].dropna()}
for df in (train, test):
    df['hour'] = pd.to_datetime(df['request_ts'], unit='s').dt.hour
    df['domain'] = df['referer'].str.replace('https://', '').str.split('/').str[0]
    df['browser'] = df['user_agent'].map(lambda ua: ua_map.get(ua, {}).get('browser', 'none'))
    df['os'] = df['user_agent'].map(lambda ua: ua_map.get(ua, {}).get('os', 'none'))

train = train.merge(labels, on='user_id')
test = test.merge(test_users, on='user_id')
train = train.merge(geo, on='geo_id', how='left').merge(ref_vec, on='referer', how='left')
test = test.merge(geo, on='geo_id', how='left').merge(ref_vec, on='referer', how='left')
print(f"Shapes after merge: train {train.shape}, test {test.shape}")

categories = ['hour', 'domain', 'browser', 'os', 'country_id', 'region_id']
components = [col for col in train.columns if col.startswith('component')]
X = train[categories + components]
y = train['target']

print("Preparing training and validation sets...")
X_train, X_values, y_train, y_values = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test[categories + components]

print("Training CatBoost model...")
model = CatBoostClassifier(iterations=1000, auto_class_weights='Balanced', cat_features=categories, verbose=50)
model.fit(X_train, y_train, eval_set=(X_values, y_values))

print("Predicting on test dataset...")
proba = model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({'user_id': test['user_id'], 'target': (proba > 0.5).astype(int)})
os.makedirs(OUTPUT_DIR, exist_ok=True)
submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), sep=';', index=False)

predict = model.predict(X_values)

metrics = {
    "Accuracy": accuracy_score(y_values, predict),
    "F1 Score": f1_score(y_values, predict),
    "ROC AUC": roc_auc_score(y_values, model.predict_proba(X_values)[:, 1])
}

lenght = 50
print("\n" + "=" * lenght)
print(f"{'Result Metrics':^{lenght}}")
print("=" * lenght)
for name, value in metrics.items():
    print(f"{name:<20}: {value}")
print("=" * lenght)
print(f"{'Result saved to':<20}: {os.path.join(OUTPUT_DIR, 'submission.csv')}")
print("=" * lenght)
