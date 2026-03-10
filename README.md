# BLENDED LEARNING
# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset containing multiple features with high dimensionality.

2.Separate the dataset into input features and target class (if present) and handle missing values if necessary.

3.Normalize or standardize the feature values so that all features contribute equally to the analysis.

4.Compute the covariance matrix of the standardized features to analyze relationships between variables.

5.Calculate the eigenvalues and eigenvectors from the covariance matrix.

6.Sort the eigenvalues in descending order and select the top principal components that capture the maximum variance.

7.Transform the original dataset by projecting it onto the selected principal components.

8.Obtain the reduced dataset with fewer dimensions while preserving most of the important information.

9.Use the reduced dataset for visualization or further machine learning tasks. 

## Program:
```
/*
Program to implement Principal Component Analysis (PCA) for dimensionality reduction on the energy data.
Developed by: Shrivarshan 
RegisterNumber:  25019111
*/
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the dataset
data = pd.read_csv('HeightsWeights.csv')


# Step 2: Display first few rows
print("First 5 rows of the dataset:")
print(data.head())


# Step 3: Select features
X = data[['Height(Inches)', 'Weight(Pounds)']]


# Step 4: Visualize original data
plt.figure(figsize=(6,5))
sns.scatterplot(x='Height(Inches)', y='Weight(Pounds)', data=data)
plt.title("Original Data Distribution")
plt.show()


# Step 5: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 6: Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# Step 7: Explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)


# Step 8: Create PCA DataFrame
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])


# Step 9: Plot PCA result
plt.figure(figsize=(6,5))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA Projection of Height and Weight")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

```

## Output:
![alt text](<Screenshot 2026-03-10 171024.png>) 
![alt text](<Screenshot 2026-03-10 171014.png>)


## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.
