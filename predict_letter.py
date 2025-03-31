import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

model = load_model('cattell16_prediction.keras')

user_input_raw = [
    2, 2, 3, 3, 3,   # A1–A5
    4, 4, 4, 4, 4,   # B1–B5
    2, 3, 3, 3, 1,   # C1–C5
    4, 5, 5, 5, 3    # D1–D5
]

df = pd.read_csv("data.csv", delimiter="\t")
df.columns = df.columns.str.replace('"', '').str.strip()
letter_blocks = {letter: [col for col in df.columns if col.startswith(letter)] for letter in ['A', 'B', 'C', 'D']}

mean_values = []
for letter in ['A', 'B', 'C', 'D']:
    cols = letter_blocks[letter][:5]
    mean_values.extend(df[cols].replace(0, np.nan).mean().round().astype(int).tolist())

user_input = np.array(user_input_raw) - 1
mean_input = np.array(mean_values) - 1
X_input = np.concatenate([user_input, mean_input]).reshape(4, 10)  # 4 litere, câte 10 feature-uri

preds = model.predict(X_input)

for i, letter_preds in enumerate(preds):
    letter_name = chr(ord('A') + i)
    print(f"\n🔮 Predicții pentru {letter_name}6–{letter_name}10:")
    for j, probas in enumerate(letter_preds):
        predicted = np.argmax(probas) + 1
        print(f"{letter_name}{j+6}: {predicted} (distribuție: {np.round(probas, 2)})")
