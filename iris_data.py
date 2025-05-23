from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
iris = load_iris()

# Features and target
X = iris.data
y = iris.target

# Create DataFrame with features and target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

df.to_csv('iris_data.csv', index=False)