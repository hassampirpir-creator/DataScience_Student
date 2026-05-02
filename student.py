import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


np.random.seed(42)
n_students = 500

data = {
    'Study_Hours': np.random.uniform(5, 30, n_students),
    'Attendance': np.random.uniform(50, 100, n_students),
    'Sleep_Hours': np.random.uniform(4, 9, n_students),
    'Prev_Grade': np.random.uniform(50, 100, n_students)
}

df = pd.DataFrame(data)


df['Exam_Score'] = (
    (df['Study_Hours'] * 1.2) + 
    (df['Attendance'] * 0.3) + 
    (df['Prev_Grade'] * 0.2) + 
    (5 * np.sin(df['Sleep_Hours'] - 7)) + 
    np.random.normal(0, 4, n_students)
)
df['Exam_Score'] = df['Exam_Score'].clip(0, 100)


df['Study_Efficiency'] = df['Exam_Score'] / df['Study_Hours']


sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(data=df, x='Study_Hours', y='Exam_Score', hue='Attendance', ax=axes[0], palette='viridis')
axes[0].set_title('Study vs Score (Colored by Attendance)')

sns.kdeplot(data=df, x='Exam_Score', fill=True, color="blue", ax=axes[1])
axes[1].set_title('Distribution of Final Scores')

sns.boxplot(x=pd.cut(df['Sleep_Hours'], bins=3), y=df['Exam_Score'], ax=axes[2])
axes[2].set_title('Exam Score by Sleep Category')
plt.show()


X = df[['Study_Hours', 'Attendance', 'Sleep_Hours', 'Prev_Grade']]
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("--- FEATURE IMPORTANCE ---")
print(feature_importance_df)
print(f"\nModel Accuracy (R²): {r2_score(y_test, model.predict(X_test)):.2f}")