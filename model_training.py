from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

df = pd.read_csv(r"C:\Users\ADMIN\Desktop\RESUMES\New folder\Final_StudentsPerformance_Internet222.csv")

pass_count = len(df[df['label'] == 1])
fail_count = len(df[df['label'] == 0])

if pass_count > fail_count:
    df = df[df['label'] == 1].sample(fail_count).append(df[df['label'] == 0])
elif fail_count > pass_count:
    df = df[df['label'] == 0].sample(pass_count).append(df[df['label'] == 1])

X = df[['gender','parental level of education','test preparation course',	'reading score',	'writing score',	'study hours',	'extracurricular activities',	'sleep hours',	'internet usage',	'physical health',	'mental health'
]]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
