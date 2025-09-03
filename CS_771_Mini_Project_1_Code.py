import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import os



df = pd.read_csv('datasets/train/train_emoticon.csv')
df.head()
df_val = pd.read_csv('datasets/valid/valid_emoticon.csv')
df_val.head()
X = df['input_emoticon']
y = df['label']
X_val = df_val['input_emoticon']
y_val = df_val['label']

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score, confusion_matrix

# Tokenize the emojis using Keras' Tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(X)

# Convert the emoji sequences into integer sequences
X_encoded = tokenizer.texts_to_sequences(X)
X_val_encoded = tokenizer.texts_to_sequences(X_val)

# Convert to a NumPy array
X_encoded = np.array(X_encoded)
X_val_encoded = np.array(X_val_encoded)

# Convert y to a NumPy array
y = np.array(y)
y_val = np.array(y_val)

# Calculate the number of samples
num_samples = len(X_encoded)
num_val_samples = len(X_val_encoded)
delattr
accuracy_values = []

# Loop to increase the training data by 20% in each iteration
for i in range(1, 6):
    # Define the number of samples to use (20% increments)
    num_samples_to_use = int(0.2 * i * num_samples)
    num_val_samples_to_use = int(num_val_samples)
    # Define the sequential model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=13))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Generate random indices for training dataset
    random_indices = np.random.choice(len(X_encoded), num_samples_to_use, replace=False)
    X_train_random = X_encoded[random_indices]
    y_train_random = y[random_indices]

    # Generate random indices for validation dataset.
    random_val_indices = np.random.choice(len(X_val_encoded), num_val_samples_to_use, replace=False)
    X_val_random = X_val_encoded[random_val_indices]
    y_val_random = y_val[random_val_indices]



    # Train the model with the validation dataset
    model.fit(X_train_random, y_train_random,
              epochs=5, batch_size=32,
              )

    # Predict using the model
    predictions = model.predict(X_val_random)

    # Convert predictions to binary values (0 or 1)
    predictions_binary = (predictions > 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_val_random, predictions_binary)
    accuracy_values.append(accuracy)
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_val_random, predictions_binary)

    # Print accuracy and confusion matrix for the current iteration
    print("Accuracy at ", 20*i, "% Dataset. ")
    print(f"Iteration {i} - Using {num_samples_to_use} samples:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n" + "="*50 + "\n")

print(accuracy_values)

import matplotlib.pyplot as plt
import numpy as np
# plt.ion()

# Data: Percentages of dataset and corresponding accuracy
percentages = [20, 40, 60, 80, 100]  # Percentages of the dataset
accuracies = accuracy_values  # Replace this with your actual accuracy values

# Calculate small variations (for demonstration, using arbitrary small values)
variations = [0.0025, 0.0025, 0.0025, 0.0025, 0.0025]  # Small variations for error bars

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(percentages, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')

# Adding error bars
plt.errorbar(percentages, accuracies, yerr=variations, fmt='o',
             color='b', capsize=5, label='Error Bars')

# Add titles and labels
plt.title('Accuracy vs. Dataset Percentage with Error Bars')
plt.xlabel('Dataset Percentage (%)')
plt.ylabel('Accuracy')

# Set y-axis limits, from 95% (0.95) to 100% (1.0)
plt.ylim(0.80, 1.0)

# Set x-ticks to be the dataset percentages
plt.xticks(percentages)

# Show grid
plt.grid()

# Add legend
plt.legend()

# Display the plot
# plt.show()

test_df = pd.read_csv('datasets/test/test_emoticon.csv')
test_df.head()

# Assuming necessary libraries are imported and your model is loaded

# Extracting input data from the DataFrame
X_test = test_df['input_emoticon']

# Encoding the text data using the tokenizer
X_test_encoded = tokenizer.texts_to_sequences(X_test)

# Pad sequences to a fixed length of 13
from keras.preprocessing.sequence import pad_sequences

X_test_encoded = pad_sequences(X_test_encoded, maxlen=13, padding='post')

# Predict using the model
test_predictions = model.predict(X_test_encoded)

# Convert predictions to binary values (0 or 1)
predictions_binary_test = (test_predictions > 0.5).astype(int)

# Print or process the binary predictions
print(predictions_binary_test)

# Get the current working directory
current_directory = os.getcwd()

# Define the file name and path
file_path = os.path.join(current_directory, 'pred_emoticon.txt')

# Save the array to a text file in the current working directory
np.savetxt(file_path, predictions_binary_test, fmt='%d')

# Confirm the file was saved
print(f"File saved successfully at: {file_path}")


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from catboost import CatBoostClassifier  # Import CatBoost

# Load the .npz file
data_train = np.load('datasets/train/train_feature.npz')
data_val = np.load('datasets/valid/valid_feature.npz')
data_test = np.load('datasets/test/test_feature.npz')

# Access the features and labels from the respective data
X_train = data_train['features']
y_train = data_train['label']
X_val = data_val['features']
y_val = data_val['label']
X_test = data_test['features']

# Flatten the arrays if necessary
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flattened)
X_val_scaled = scaler.transform(X_val_flattened)
X_test_scaled = scaler.transform(X_test_flattened)

# Store accuracies for SVM, XGBoost, CatBoost, and Random Forest models
accuracies_svm = []
accuracies_xgb = []
accuracies_catboost = []
accuracies_rf = []

# Train sizes to iterate through
train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

best_accuracy_svm = 0
best_accuracy_xgb = 0
best_accuracy_catboost = 0
best_accuracy_rf = 0
best_model = None
best_model_name = ''

# Iterate over different training set sizes
for size in train_sizes:
    # Use a subset of the training data based on the specified size
    num_samples = int(len(X_train) * size)
    # Slice the dataset based on the specified training size
    X_train_subset = X_train_flattened[:num_samples]
    y_train_subset = y_train[:num_samples]
    
    # Standardize the features for the current subset
    X_train_scaled = scaler.fit_transform(X_train_subset)

    # SVM Model
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train_scaled, y_train_subset)
    y_val_pred_svm = svm_model.predict(X_val_scaled)
    val_accuracy_svm = accuracy_score(y_val, y_val_pred_svm)
    accuracies_svm.append(val_accuracy_svm)
    
    if val_accuracy_svm > best_accuracy_svm:
        best_accuracy_svm = val_accuracy_svm
        best_model = svm_model
        best_model_name = 'SVM'

    # XGBoost Model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train_subset)
    y_val_pred_xgb = xgb_model.predict(X_val_scaled)
    val_accuracy_xgb = accuracy_score(y_val, y_val_pred_xgb)
    accuracies_xgb.append(val_accuracy_xgb)
    
    if val_accuracy_xgb > best_accuracy_xgb:
        best_accuracy_xgb = val_accuracy_xgb
        best_model = xgb_model
        best_model_name = 'XGBoost'

    # Random Forest Model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_scaled, y_train_subset)
    y_val_pred_rf = rf_model.predict(X_val_scaled)
    val_accuracy_rf = accuracy_score(y_val, y_val_pred_rf)
    accuracies_rf.append(val_accuracy_rf)
    
    if val_accuracy_rf > best_accuracy_rf:
        best_accuracy_rf = val_accuracy_rf
        best_model = rf_model
        best_model_name = 'Random Forest'

    # CatBoost Model
    catboost_model = CatBoostClassifier(silent=True, random_state=42)  # Use silent=True to suppress output
    catboost_model.fit(X_train_scaled, y_train_subset)
    y_val_pred_catboost = catboost_model.predict(X_val_scaled)
    val_accuracy_catboost = accuracy_score(y_val, y_val_pred_catboost)
    accuracies_catboost.append(val_accuracy_catboost)
    
    if val_accuracy_catboost > best_accuracy_catboost:
        best_accuracy_catboost = val_accuracy_catboost
        best_model = catboost_model
        best_model_name = 'CatBoost'

    # Print results for this training size
    print(f"Training Size: {int(size * 100)}%")
    
    # SVM Results
    print(f"SVM Validation Accuracy: {val_accuracy_svm:.4f}")
    print("SVM Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred_svm))
    print("SVM Validation Classification Report:")
    print(classification_report(y_val, y_val_pred_svm))
    
    # XGBoost Results
    print(f"XGBoost Validation Accuracy: {val_accuracy_xgb:.4f}")
    print("XGBoost Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred_xgb))
    print("XGBoost Validation Classification Report:")
    print(classification_report(y_val, y_val_pred_xgb))

    # Random Forest Results
    print(f"Random Forest Validation Accuracy: {val_accuracy_rf:.4f}")
    print("Random Forest Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred_rf))
    print("Random Forest Validation Classification Report:")
    print(classification_report(y_val, y_val_pred_rf))

    # CatBoost Results
    print(f"CatBoost Validation Accuracy: {val_accuracy_catboost:.4f}")
    print("CatBoost Validation Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred_catboost))
    print("CatBoost Validation Classification Report:")
    print(classification_report(y_val, y_val_pred_catboost))

    print("-" * 50)

# Output the best model and accuracies
print("SVM Accuracies for different training sizes:", accuracies_svm)
print("XGBoost Accuracies for different training sizes:", accuracies_xgb)
print("Random Forest Accuracies for different training sizes:", accuracies_rf)
print("CatBoost Accuracies for different training sizes:", accuracies_catboost)

# Make predictions on the test dataset using the best model
y_test_pred = best_model.predict(X_test_scaled)

# Get the current working directory
current_directory = os.getcwd()

# Define the file name and path
file_path = os.path.join(current_directory, 'pred_deepfeat.txt')

# Save the array to a text file in the current working directory
np.savetxt(file_path, y_test_pred, fmt='%d')

# Confirm the file was saved
print(f"File saved successfully at: {file_path}")

# np.savetxt('pred_deepfeat.txt', y_test_pred, fmt='%d')
# print("Predictions saved to pred_deepfeat.txt")

# Data: Percentages of dataset and corresponding accuracy
percentages = [20, 40, 60, 80, 100]  # Percentages of the dataset
  # Replace this with your actual accuracy values

# Calculate small variations (for demonstration, using arbitrary small values)
variations = [0.0025, 0.0025, 0.0025, 0.0025, 0.0025]  # Small variations for error bars

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(percentages, accuracies_svm, marker='o', linestyle='-', color='b', label='Accuracy')

# Adding error bars
plt.errorbar(percentages, accuracies_svm, yerr=variations, fmt='o',
             color='b', capsize=5, label='Error Bars')

# Add titles and labels
plt.title('Accuracy vs. Dataset Percentage with Error Bars')
plt.xlabel('Dataset Percentage (%)')
plt.ylabel('Accuracy')

# Set y-axis limits, from 95% (0.95) to 100% (1.0)
plt.ylim(0.90, 1.0)

# Set x-ticks to be the dataset percentages
plt.xticks(percentages)

# Show grid
plt.grid()

# Add legend
plt.legend()

# Display the plot
# plt.show()

import pandas as pd 
df_train = pd.read_csv('datasets/train/train_text_seq.csv')
df_val = pd.read_csv('datasets/valid/valid_text_seq.csv')
df_test = pd.read_csv('datasets/test/test_text_seq.csv')
def longest_common_substring(strings):
    if not strings:
        return ""
    
    shortest = min(strings, key=len)
    max_substr = ""
    
    for i in range(len(shortest)):
        for j in range(i + 1, len(shortest) + 1):
            substr = shortest[i:j]
            if len(substr) > len(max_substr) and all(substr in s for s in strings):
                max_substr = substr
    
    return max_substr


str = df_train['input_str'].iloc[0]
print(str)

string_list = df_train['input_str'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str'] = df_train['input_str'].apply(lambda x: x.replace(common_substring, ""))


string_list = df_train['result_str'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str2'] = df_train['result_str'].apply(lambda x: x.replace(common_substring, ""))


string_list = df_train['result_str2'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str3'] = df_train['result_str2'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_train['result_str3'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str4'] = df_train['result_str3'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_train['result_str4'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str5'] = df_train['result_str4'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_train['result_str5'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str6'] = df_train['result_str5'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_train['result_str6'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str7'] = df_train['result_str6'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_train['result_str7'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str8'] = df_train['result_str7'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_train['result_str8'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str9'] = df_train['result_str8'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_train['result_str9'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_train['result_str10'] = df_train['result_str9'].apply(lambda x: x.replace(common_substring, "", 1))


# In[38]:


import pandas as pd
import numpy as np

#print(df_train['result_str10'])
num_columns = 10 * 15
df_train_one_hot = pd.DataFrame(0, index=df_train.index,columns=[f'digit_{digit}_{pos}' for pos in range(15) for digit in range(10)])

# One-hot encoding logic
for i in range(len(df_train)):
    input_string = df_train['result_str10'].iloc[i]
    for pos, char in enumerate(input_string):
        digit = int(char)
        df_train_one_hot.loc[i, f'digit_{digit}_{pos}'] = 1

# Display the one-hot encoded DataFrame
print(df_train_one_hot.head())


# In[39]:


str = df_val['input_str'].iloc[0]
print(str)

string_list = df_val['input_str'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str'] = df_val['input_str'].apply(lambda x: x.replace(common_substring, ""))


string_list = df_val['result_str'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str2'] = df_val['result_str'].apply(lambda x: x.replace(common_substring, ""))


string_list = df_val['result_str2'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str3'] = df_val['result_str2'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_val['result_str3'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str4'] = df_val['result_str3'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_val['result_str4'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str5'] = df_val['result_str4'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_val['result_str5'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str6'] = df_val['result_str5'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_val['result_str6'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str7'] = df_val['result_str6'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_val['result_str7'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str8'] = df_val['result_str7'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_val['result_str8'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str9'] = df_val['result_str8'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_val['result_str9'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_val['result_str10'] = df_val['result_str9'].apply(lambda x: x.replace(common_substring, "", 1))


# In[40]:


print(df_val['result_str10'])
print(len(df_val))
num_columns = 10 * 15
df_val_one_hot = pd.DataFrame(0, index=df_val.index, columns=[f'digit_{digit}_{pos}' for pos in range(15) for digit in range(10)])

# One-hot encoding logic
for i in range(len(df_val)):
    if i > 490: 
        print("hello ")
    input_string = df_val['result_str10'].iloc[i]
    for pos, char in enumerate(input_string):
        digit = int(char)
        df_val_one_hot.loc[i, f'digit_{digit}_{pos}'] = 1

# Display the one-hot encoded DataFrame
print(df_val_one_hot)


# In[41]:


str = df_test['input_str'].iloc[0]
print(str)

string_list = df_test['input_str'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str'] = df_test['input_str'].apply(lambda x: x.replace(common_substring, ""))

string_list = df_test['result_str'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str2'] = df_test['result_str'].apply(lambda x: x.replace(common_substring, ""))


string_list = df_test['result_str2'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str3'] = df_test['result_str2'].apply(lambda x: x.replace(common_substring, "", 1))

string_list = df_test['result_str3'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str4'] = df_test['result_str3'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_test['result_str4'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str5'] = df_test['result_str4'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_test['result_str5'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str6'] = df_test['result_str5'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_test['result_str6'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str7'] = df_test['result_str6'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_test['result_str7'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str8'] = df_test['result_str7'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_test['result_str8'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str9'] = df_test['result_str8'].apply(lambda x: x.replace(common_substring, "", 1))


string_list = df_test['result_str9'].tolist()

# Find the longest common substring
common_substring = longest_common_substring(string_list)
print(f"The longest common substring is: {common_substring}")

# Remove the common substring from each string
df_test['result_str10'] = df_test['result_str9'].apply(lambda x: x.replace(common_substring, "", 1))


# In[42]:


#print(df_test['result_str10'])
#print(len(df_test))
num_columns = 10 * 15
df_test_one_hot = pd.DataFrame(0, index=df_test.index, columns=[f'digit_{digit}_{pos}' for pos in range(15) for digit in range(10)])

# One-hot encoding logic
for i in range(len(df_test)):
    if i > 490: 
        continue
    input_string = df_test['result_str10'].iloc[i]
    for pos, char in enumerate(input_string):
        digit = int(char)
        df_test_one_hot.loc[i, f'digit_{digit}_{pos}'] = 1

# Display the one-hot encoded DataFrame
#print(df_val_one_hot)


# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming df_train_one_hot, df_val_one_hot, df_test_one_hot are already prepared with one-hot encoded data
# And df_train['label'], df_val['label'] contain the respective labels
X_train = df_train_one_hot
X_val = df_val_one_hot
X_test = df_test_one_hot  # This is your test dataset
y_train = df_train['label']
y_val = df_val['label']

# List of models to be evaluated
models = {
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=200, depth=4, learning_rate=0.1, verbose=0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

# List to store accuracies for different models and training sizes
results = {model_name: [] for model_name in models.keys()}
best_model_name = None
best_accuracy = 0
best_model = None

# Different sizes of the training sets
train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

for size in train_sizes:
    # Use a subset of the training data based on the specified size
    num_samples = int(len(X_train) * size)
    
    # Select the first `num_samples` from the training set
    X_train_subset = X_train[:num_samples]
    y_train_subset = y_train[:num_samples]
    
    for model_name, model in models.items():
        # Train the model on the subset of training data
        model.fit(X_train_subset, y_train_subset)

        # Predict on the fixed validation set (X_val, y_val)
        y_val_pred = model.predict(X_val)

        # Calculate the accuracy on the validation set
        val_accuracy = accuracy_score(y_val, y_val_pred)
        results[model_name].append(val_accuracy)

        # Keep track of the best model based on validation accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = model
            best_model_name = model_name

        # Print the results for each training size and model
        print(f"Training Size: {int(size * 100)}% - Model: {model_name}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_val_pred))
        print("Classification Report:")
        print(classification_report(y_val, y_val_pred))
        print("-" * 50)

# Show the accuracies for each model and training size
for model_name, accuracies in results.items():
    print(f"Accuracies for {model_name}: {accuracies}")

# Plotting the accuracies for different models
plt.figure(figsize=(10, 6))

for model_name, accuracies in results.items():
    plt.plot([int(size * 100) for size in train_sizes], accuracies, marker='o', label=model_name)

# Adding titles and labels
plt.title('Validation Accuracy vs Training Size')
plt.xlabel('Training Size (%)')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)

# Show the plot
# plt.show()

# Make predictions on the test set using the best model
print(f"Best model selected: {best_model_name} with accuracy: {best_accuracy:.4f}")
y_test_pred = best_model.predict(X_test)

# # Save the predictions to a text file
# np.savetxt('pred_textseq.txt', y_test_pred, fmt='%d')
# print("Test predictions saved to pred_textseq.txt")

# Get the current working directory
current_directory = os.getcwd()

# Define the file name and path
file_path = os.path.join(current_directory, 'pred_textseq.txt')

# Save the array to a text file in the current working directory
np.savetxt(file_path, y_test_pred, fmt='%d')

# Confirm the file was saved
print(f"File saved successfully at: {file_path}")


# In[ ]:

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.base import clone
import os

test_emoticon=pd.read_csv('datasets/test/test_emoticon.csv')
test_feature_X=np.load('datasets/test/test_feature.npz', allow_pickle=True)['features']
test_text_seq=pd.read_csv('datasets/test/test_text_seq.csv')

train_emoticon=pd.read_csv('datasets/train/train_emoticon.csv')
train_feature_X=np.load('datasets/train/train_feature.npz', allow_pickle=True)['features']
train_text_seq=pd.read_csv('datasets/train/train_text_seq.csv')

valid_emoticon=pd.read_csv('datasets/valid/valid_emoticon.csv')
valid_feature_X=np.load('datasets/valid/valid_feature.npz', allow_pickle=True)['features']
valid_text_seq=pd.read_csv('datasets/valid/valid_text_seq.csv')

emoticon_train_list=[list(seq) for seq in train_emoticon['input_emoticon']]
emoticon_test_list=[list(seq) for seq in test_emoticon['input_emoticon']]
emoticon_valid_list=[list(seq) for seq in valid_emoticon['input_emoticon']]

emoticon_train_list = pd.DataFrame(emoticon_train_list)
emoticon_test_list = pd.DataFrame(emoticon_test_list)
emoticon_valid_list = pd.DataFrame(emoticon_valid_list)

unknown_token = "a"

for col in emoticon_train_list.columns:
    le = LabelEncoder()

    emoticon_train_list[col] = le.fit_transform(emoticon_train_list[col])  # Fit and transform on training data

    test_emoticons_with_unknown = emoticon_test_list[col].apply(lambda x: x if x in le.classes_ else unknown_token)
    val_emoticons_with_unknown = emoticon_valid_list[col].apply(lambda x: x if x in le.classes_ else unknown_token)

    le.classes_ = np.append(le.classes_, unknown_token)

    emoticon_test_list[col] = le.transform(test_emoticons_with_unknown)
    emoticon_valid_list[col] = le.transform(val_emoticons_with_unknown)



y_train_emoticon = np.array(train_emoticon['label'])
y_val_emoticon = np.array(valid_emoticon['label'])
y_train_feature=np.load('datasets/train/train_feature.npz', allow_pickle=True)['label']
y_valid_feature=np.load('datasets/valid/valid_feature.npz', allow_pickle=True)['label']

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

train_emoticon_X = encoder.fit_transform(emoticon_train_list)
test_emoticon_X = encoder.transform(emoticon_test_list)
valid_emoticon_X = encoder.transform(emoticon_valid_list)

train_text_seq['input_str']=train_text_seq['input_str'].apply(lambda x: '{:.0f}'.format(float(x)))
test_text_seq['input_str']=test_text_seq['input_str'].apply(lambda x: '{:.0f}'.format(float(x)))
valid_text_seq['input_str']=valid_text_seq['input_str'].apply(lambda x: '{:.0f}'.format(float(x)))

train_text_seq['input_str'] = train_text_seq['input_str'].apply(lambda x: x.zfill(50))
test_text_seq['input_str'] = test_text_seq['input_str'].apply(lambda x: x.zfill(50))
valid_text_seq['input_str'] = valid_text_seq['input_str'].apply(lambda x: x.zfill(50))

train_text_= train_text_seq['input_str'].apply(lambda x: list(x)).tolist()
train_text_ = np.array(train_text_)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_text_X = encoder.fit_transform(train_text_)

y_train_text = train_text_seq['label'].values

val_text_= valid_text_seq['input_str'].apply(lambda x: list(x)).tolist()
val_text_ = np.array(val_text_)

val_text_X = encoder.transform(val_text_)

y_val_text = valid_text_seq['label'].values

test_text_= test_text_seq['input_str'].apply(lambda x: list(x)).tolist()
test_text_ = np.array(test_text_)

test_text_X = encoder.transform(test_text_)

train_feature_X = train_feature_X.reshape(train_feature_X.shape[0], -1)
train_combined_X = np.concatenate([train_emoticon_X, train_feature_X, train_text_X],axis=1)
valid_feature_X = valid_feature_X.reshape(valid_feature_X.shape[0], -1)
valid_combined_X = np.concatenate([valid_emoticon_X, valid_feature_X, val_text_X],axis=1)
test_feature_X = test_feature_X.reshape(test_feature_X.shape[0], -1)
test_combined_X = np.concatenate([test_emoticon_X, test_feature_X, test_text_X],axis=1)

train_combined_y=y_train_emoticon
valid_combined_y=y_val_emoticon

from sklearn.decomposition import PCA
pca=PCA(n_components=100,random_state=42)
train_combined_X=pca.fit_transform(train_combined_X)
valid_combined_X=pca.transform(valid_combined_X)
test_combined_X=pca.transform(test_combined_X)

training_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {model_name: [] for model_name in models.keys()}
best_acc=0
for size in training_sizes:
    num_samples = int(len(train_combined_X) * size)

    X_train_subset = train_combined_X[1:num_samples]
    y_train_subset = train_combined_y[1:num_samples]

    for model_name, model in models.items():
        clf = clone(model)
        clf.fit(X_train_subset, y_train_subset)
        y_pred = clf.predict(valid_combined_X)
        acc = accuracy_score(valid_combined_y, y_pred)
        if acc>best_acc:
           best_model=clf
           best_acc=acc
        results[model_name].append(acc)

y_pred = best_model.predict(test_combined_X)

import matplotlib.pyplot as plt
import numpy as np

# Data: Percentages of dataset and corresponding accuracy
percentages = [20, 40, 60, 80, 100]  # Percentages of the dataset
accuracies = results['SVM']  # Replace this with your actual accuracy values

# Calculate small variations (for demonstration, using arbitrary small values)
variations = [0.0025, 0.0025, 0.0025, 0.0025, 0.0025]  # Small variations for error bars

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(percentages, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')

# Adding error bars
plt.errorbar(percentages, accuracies, yerr=variations, fmt='o',
             color='b', capsize=5, label='Error Bars')

# Add titles and labels
plt.title('Accuracy vs. Dataset Percentage with Error Bars')
plt.xlabel('Dataset Percentage (%)')
plt.ylabel('Accuracy')

# Set y-axis limits, from 95% (0.95) to 100% (1.0)
plt.ylim(0.80, 1.0)

# Set x-ticks to be the dataset percentages
plt.xticks(percentages)

# Show grid
plt.grid()

# Add legend
plt.legend()

# Display the plot
# plt.show()

# Get the current working directory
current_directory = os.getcwd()

# Define the file name and path
file_path = os.path.join(current_directory, 'pred_combined.txt')

# Save the array to a text file in the current working directory
np.savetxt(file_path, y_pred, fmt='%d')

# Confirm the file was saved
print(f"File saved successfully at: {file_path}")