#1 and 2
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

#3 question

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classifier(df):
    features = ["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]
    X = df[features]
    y = df['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    df['Predicted Category'] = classifier.predict(X)
    return df


# Load the dataset into a pandas DataFrame
df = pd.read_excel("C:\Users\prane\Desktop\Lab Session1 Data (1).xlsx")

# Create the 'Category' column based on the payment amount
df['Category'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Run the classifier function
df = classifier(df)

# Print the relevant columns
print(df[['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)', 'Category', 'Predicted Category']])


#4 question

import pandas as pd
import statistics
import matplotlib.pyplot as plt

excel_file_path = '"C:\Users\prane\Desktop\Lab Session1 Data (1).xlsx"
df = pd.read_excel(excel_file_path, sheet_name='IRCTC Stock Price')

price_mean = statistics.mean(df['Price'])
price_variance = statistics.variance(df['Price'])
print(f"Mean of Price: {price_mean}\n")
print(f"Variance of Price: {price_variance}\n")

wednesday_data = df[df['Day'] == 'Wed']
wednesday_mean = statistics.mean(wednesday_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")
print(f"Sample Mean of Price on Wednesdays: {wednesday_mean}\n")


april_data = df[df['Month'] == 'Apr']
april_mean = statistics.mean(april_data['Price'])
print(f"Population Mean of Price: {price_mean}\n")
print(f"Sample Mean of Price in April: {april_mean}\n")


loss_probability = len(df[df['Chg%'] < 0]) / len(df)
print(f"Probability of making a loss: {loss_probability}\n")
wednesday_profit_probability = len(wednesday_data[wednesday_data['Chg%'] > 0]) / len(wednesday_data)
print(f"Probability of making a profit on Wednesday: {wednesday_profit_probability}\n")
conditional_profit_probability = wednesday_profit_probability / loss_probability
print(f"Conditional Probability of making profit, given today is Wednesday: {conditional_profit_probability}\n")
day=['Mon','Tue','Wed','Thu','Fri']
day1=[]
chg1=[]
for i in day:
    for j in range(2,len(df['Day'])):
        if i==df.loc[j,'Day']:
            day1.append(i)
            chg1.append(df.loc[j,'Chg%'])
plt.scatter(day1, chg1)
plt.xlabel('Day of the Week')
plt.ylabel('Chg%')
plt.title('Scatter plot of Chg% against the day of the week')
plt.show()


