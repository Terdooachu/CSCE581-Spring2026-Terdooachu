import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


df = pd.read_excel("data/DataSample-WeightHeight.xlsx")


df = df.dropna()

# Renames the  columns
df = df.rename(columns={
    'Height (cm)': 'height',
    'Weight (kg)': 'weight',
    'BMI (numeric)': 'bmi',
    'BMI Category = C1, C2, C3, C4': 'bmi_class'
})

# -------------------------
# TASK 1: Prediction of weight from height
# -------------------------
X = df[['height']]
y = df['weight']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("TASK 1: Prediction (Weight from Height)")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print()

# -------------------------
# TASK 2: Classifies the BMI category from height
# -------------------------
Xc = df[['height']]
yc = df['bmi_class']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(Xc, yc, test_size=0.2)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_c, y_train_c)

yc_pred = clf.predict(X_test_c)

print("TASK 2: Classification (BMI Category from Height)")
print("Accuracy:", accuracy_score(y_test_c, yc_pred))
print(classification_report(y_test_c, yc_pred))

# -------------------------
# TASK 3: Classifies the BMI category from height AND weight
# -------------------------
X_hw = df[['height', 'weight']]
y_hw = df['bmi_class']

X_train_hw, X_test_hw, y_train_hw, y_test_hw = train_test_split(
    X_hw, y_hw, test_size=0.2
)

clf_hw = LogisticRegression(max_iter=1000)
clf_hw.fit(X_train_hw, y_train_hw)

y_hw_pred = clf_hw.predict(X_test_hw)

print("TASK 3: Classification (BMI Category from Height and Weight)")
print("Accuracy:", accuracy_score(y_test_hw, y_hw_pred))
print(classification_report(y_test_hw, y_hw_pred, zero_division=0))
