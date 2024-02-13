import pandas as pd
import numpy as np
 
data = pd.read_excel("C:\\Users\\prane\\Downloads\\Lab Session1 Data.xlsx", sheet_name="Purchase data")
print(data)
 
A = data[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
C = data[['Payment (Rs)']].values
 
print("Matrix A:\n", A)
print("Matrix C:\n", C)
 
dimensionality = A.shape[1]
print("Dimensionality of the vector space:", dimensionality)
 
num_vectors = A.shape[0]
print("Number of vectors in the vector space:", num_vectors)
 
rank_A = np.linalg.matrix_rank(A)
print("Rank of Matrix A:", rank_A)
 
A_pseudo_inv = np.linalg.pinv(A)
cost_per_product = np.dot(A_pseudo_inv, C)
print("Cost of each product available for sale:", cost_per_product)
 
model_vector_X = np.dot(A_pseudo_inv, C)
print("Model vector X for predicting product costs:", model_vector_X)
