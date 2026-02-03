import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# --- Step 1: Load Dataset ---
data = pd.read_csv("sleep.csv")
data = data.dropna()

# --- Step 2: Map Sleep Quality ---
def sleep_quality_label(x):
    if x <= 4:
        return 0   # Poor
    elif x <= 7:
        return 1   # Average
    else:
        return 2   # Good

data['Sleep Quality'] = data['Quality of Sleep'].apply(sleep_quality_label)

# --- Step 3: Prepare Features and Labels ---
X = data[['Sleep Duration', 'Stress Level', 'Physical Activity Level']]
y = data['Sleep Quality']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 4: Train Model ---
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model Accuracy:", model.score(X_test, y_test))

# --- Step 5: Get User Input ---
sleep = float(input("Sleep duration (hours): "))
stress = int(input("Stress level (0-10): "))
activity = float(input("Physical activity level: "))

input_data = pd.DataFrame([[sleep, stress, activity]],
    columns=['Sleep Duration', 'Stress Level', 'Physical Activity Level'])

prediction = model.predict(input_data)[0]

# --- Step 6: Show Prediction ---
quality_dict = {0: "POOR 😞", 1: "AVERAGE 😐", 2: "GOOD 😴"}
print("Sleep Quality:", quality_dict[prediction])

if stress > 6:
    print("Tip: Try meditation or relaxation")
if activity < 30:
    print("Tip: Increase daily physical activity")

# --- Step 7: Visualization (Matplotlib only) ---

# 1️⃣ Sleep Quality Distribution
plt.figure()
plt.hist(data['Sleep Quality'])
plt.title("Distribution of Sleep Quality")
plt.xlabel("Sleep Quality")
plt.ylabel("Count")
plt.show()

# 2️⃣ Feature Ranges
plt.figure()
plt.boxplot([
    data['Sleep Duration'],
    data['Stress Level'],
    data['Physical Activity Level']
])
plt.xticks([1, 2, 3], ['Sleep Duration', 'Stress Level', 'Physical Activity'])
plt.title("Feature Ranges")
plt.show()

# 3️⃣ User Input vs Dataset
plt.figure()
plt.scatter(data['Sleep Duration'], data['Stress Level'], alpha=0.5)
plt.scatter(sleep, stress, s=150)
plt.title("Your Input vs Dataset")
plt.xlabel("Sleep Duration")
plt.ylabel("Stress Level")
plt.show()
