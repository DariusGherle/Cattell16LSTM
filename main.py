import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Input, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
#from keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# 1. Încarcă fișierul
df = pd.read_csv("data.csv", delimiter='\t')
df.columns = df.columns.str.replace('"', '').str.strip()

# 2. Selectăm doar întrebările din literele A–D
question_cols = [col for col in df.columns if col[0] in ['A', 'B', 'C', 'D'] and col[1:].isdigit()]
print("Coloane selectate:", question_cols)

# 3. Convertim la întreg și înlocuim 0 cu NaN
data = df[question_cols].apply(pd.to_numeric, errors='coerce')
data.replace(0, np.nan, inplace=True)

# 4. Eliminăm rândurile incomplete
data = data.dropna()
print("Rânduri rămase după curățare:", len(data))

# 5. Prelucrăm pe litere
letter_blocks = {letter: [col for col in data.columns if col.startswith(letter)] for letter in ['A', 'B', 'C', 'D']}

X_user, X_mean, y_true = [], [], []

for letter, cols in letter_blocks.items():
    user_input = data[cols[:5]].values - 1
    mean_input = data[cols[:5]].mean().values - 1
    targets = data[cols[5:10]].values - 1
    mean_replicated = np.tile(mean_input, (user_input.shape[0], 1))
    X_user.append(np.concatenate([user_input, mean_replicated], axis=1))
    y_true.append(targets)

X_user = np.vstack(X_user)
y_true = np.vstack(y_true)

X = X_user
y = y_true

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model optimizat
inputs = Input(shape=(10,), dtype='int32')
embedding = Embedding(input_dim=5, output_dim=16)(inputs)
lstm_out = LSTM(96)(embedding)
lstm_out = Dropout(0.3)(lstm_out)

outputs = []
for i in range(5):
    dense1 = Dense(64, activation='relu')(lstm_out)
    dense2 = Dense(32, activation='relu')(dense1)
    out = Dense(5, activation='softmax', name=f'letter_{i+1}')(dense2)
    outputs.append(out)

model = Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss={f'letter_{i+1}': 'sparse_categorical_crossentropy' for i in range(5)},
    metrics={f'letter_{i+1}': 'accuracy' for i in range(5)}
)
model.summary()

# 7. Pregătim targeturile
y_train_split = [y_train[:, i] for i in range(5)]
y_test_split = [y_test[:, i] for i in range(5)]

# 8. Early stopping
#early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 9. Antrenare
model.fit(
    X_train,
    y_train_split,
    validation_data=(X_test, y_test_split),
    epochs=5,
    batch_size=32,
    #callbacks=[early_stop]
)

# 10. Salvare model
keras.saving.save_model(model, 'cattell16_prediction.keras')
