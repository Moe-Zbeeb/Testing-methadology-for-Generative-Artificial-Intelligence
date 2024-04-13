from sklearn.svm import OneClassSVM
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

file_p = 'dataset.csv'
data = pd.read_csv(file_p)
data 
clf = OneClassSVM(kernel='linear', nu=0.5)
clf.fit(data)  
new_p = 'output.csv'
output_data = pd.read_csv(new_p)
output_data
predictions = clf.predict(output_data)

output_data['prediction'] = predictions
num_inliers_data = np.sum(predictions == 1)

percentage_inliers_data = (num_inliers_data / len(predictions)) * 100


print(f"Percentage of inliers for the output data: {percentage_inliers_data:.2f}%")   
random_20_percent = data.sample(frac=0.2, random_state=42)  # Set random_state for reproducibility

# Remove the selected 20% from the original DataFrame
data_2 = data.drop(random_20_percent.index)

clf1 = OneClassSVM(kernel='linear', nu=0.5)
clf1.fit(data)

predictions1 = clf1.predict(data_2)
data_2['prediction'] = predictions1
num_inliers_out = np.sum(predictions1 == 1)
percentage_inliers_out = (num_inliers_out / len(predictions1)) * 100
print(f"Percentage of inliers for the data it self: {percentage_inliers_out:.2f}%")  
def res(data,new_data,nu1):
  clf = OneClassSVM(kernel='linear', nu=nu1)
  clf.fit(data)

  predictions = clf.predict(new_data)
  new_data['prediction'] = predictions

  pca = PCA(n_components=2)
  new_data_reduced = pca.fit_transform(new_data)

  num_inliers = np.sum(predictions == 1)
  percentage_inliers = (num_inliers / len(predictions)) * 100

  return percentage_inliers  
nu = [0.005,0.01,0.05,0.1,0.15,0.2,0.25]
percent = []
for i in range(len(nu)):
  output_data = pd.read_csv(new_p)
  a = res(data,output_data,nu[i])
  percent.append(a)   
plt.plot(nu, percent, marker='o', linestyle='-')
plt.xlabel('nu')
plt.ylabel('Percent')
plt.title('Percent vs nu')
plt.grid(True)
plt.show()