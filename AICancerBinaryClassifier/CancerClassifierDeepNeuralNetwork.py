""""
    Lung Cancer Binary Classifier using miRNA Expression
    This program reads and assigns a simple "1" or "0" binary to miRNA samples
    taken from patients with lung cancer, using datasets acquired from the
    open access Cancer Genome Atlas, specifying whether the sample came from a
    patient with cancer or a non-cancer sample.

    Author: Josh Stanton
    Date: November 29th, 2024
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load the data
data = pd.read_parquet('complete_data.parquet')

"""  TRAINING BATCH #1
 The first set of training simply puts the entire set of data into the training model 
 and assesses every single miRNA across the board. This should have lower accuracy due 
 to the wide ranging variance in the datasets. 

"""
X = data.drop(['cancer'], axis=1)  # Remove 'cancer' column to allow learning
y = data['cancer'] # Indicates the correct

# Data pre-processing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Calculates the mean and standard deviation
                                   # of each miRNA

# Split into training and testing sets of even distribution
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

"""
    Create the deep neural network model, beginning with 256 nodes and gradually training
    layers with fewer nodes, ultimately resulting in the binary classification done using the
    sigmoid node.
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

"""
Compiles the model:
 The algorithm uses the adam optimizer for gradient descent,
 using the binary_crossentropy as a loss function and uses 
 prediction accuracy as the guiding performance metric. 
"""
model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])

"""
    Run the TensorFlow Training Function:
    the model is trained using the predefined testing sets of the labelled and
    unlabeled data, running for 200 epochs (iterations), with a batch size of 32
    specifying that every 32 samples leads to an update in learning. The validation
    split is also 50% to match the distribution of the previous split of test and 
    training data.  
"""
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Make predictions
predictions = model.predict(X_test)


"""  TRAINING BATCH #2
    This batch of training focuses on the miRNA identified in my thesis as established 
    miRNA of interest, yielding higher expression in lung cancer patients versus noncancerous patients.

"""

#
selected_mirnas = ['hsa-mir-181c', 'hsa-mir-500a', 'hsa-mir-99a', 'hsa-mir-10b']
X = data[selected_mirnas]
y = data['cancer']

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
"""
    Create the deep neural network model, beginning with 256 nodes and gradually training
    layers with fewer nodes, ultimately resulting in the binary classification done using the
    sigmoid node.
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

"""
    Compiles the model:
    The algorithm uses the adamax optimizer for gradient descent,
    using the binary_crossentropy as a loss function and uses 
    prediction accuracy as the guiding performance metric. 
"""
model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])


"""
    Run the TensorFlow Training Function:
    the model is trained using the predefined testing sets of the labelled and
    unlabeled data, running for 200 epochs (iterations), with a batch size of 32
    specifying that every 32 samples leads to an update in learning. The validation
    split is also 50% to match the distribution of the previous split of test and 
    training data.  
"""
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Make predictions
predictions = model.predict(X_test)


"""  TRAINING BATCH #3
    This batch of training focuses on exploring the miRNA "families" hypothesis
    that was generated at the end of the master's thesis, positing that perhaps
    miRNA dysregulation in cancer is perhaps affecting miRNA "families" (miRNA
    that have the same ID number but different suffixes). 
"""

family_mirnas = ['hsa-let-7a-1', 'hsa-let-7a-2', 'hsa-let-7a-3', 'hsa-let-7b','hsa-let-7c','hsa-let-7d', 'hsa-let-7e',
                   'hsa-let-7f-1', 'hsa-let-7f-2', 'hsa-let-7g', 'hsa-let-7i', 'hsa-mir-130a' , 'hsa-mir-130b',
                   'hsa-mir-30a', 'hsa-mir-30b', 'hsa-mir-30c-1', 'hsa-mir-30c-2', 'hsa-mir-30d', 'hsa-mir-30e',
                   'hsa-mir-323a', 'hsa-mir-323b', 'hsa-mir-181a-1', 'hsa-mir-181a-2', 'hsa-mir-181b-1',
                   'hsa-mir-181b-2', 'hsa-mir-181c', 'hsa-mir-181d','hsa-mir-500a', 'hsa-mir-500b', 'hsa-mir-10a',
                   'hsa-mir-10b',  'hsa-mir-99a', 'hsa-mir-99b']

X = data[family_mirnas]
y = data['cancer']

# Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)

"""
    Create the deep neural network model, beginning with 256 nodes and gradually training
    layers with fewer nodes, ultimately resulting in the binary classification done using the
    sigmoid node.
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

"""
    Compiles the model:
    The algorithm uses the adamax optimizer for gradient descent,
    using the binary_crossentropy as a loss function and uses 
    prediction accuracy as the guiding performance metric. 
"""
# Compile the model
model.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])


"""
    Run the TensorFlow Training Function:
    the model is trained using the predefined testing sets of the labelled and
    unlabeled data, running for 200 epochs (iterations), with a batch size of 32
    specifying that every 32 samples leads to an update in learning. The validation
    split is also 50% to match the distribution of the previous split of test and 
    training data.  
"""

history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.5)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Make predictions
predictions = model.predict(X_test)


