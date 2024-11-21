import mlflow
import mlflow.tensorflow
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# تحميل البيانات
file_path = "Transformed Data Set - Sheet1.csv"
data = pd.read_csv(file_path)

# تحويل البيانات باستخدام LabelEncoder
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# فصل الميزات (X) والعلامات (y)
X = data.drop('Gender', axis=1)
y = data['Gender']

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تجهيز البيانات باستخدام StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# تعيين URI لخادم التتبع الخاص بـ MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5001")

# تسجيل MLflow
mlflow.tensorflow.autolog()
mlflow.set_experiment("Gender Classification Experiment")

# بناء نموذج الشبكة العصبية
model = Sequential()

# إضافة الطبقات
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# عرض ملخص النموذج
model.summary()

# تجميع النموذج
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# تدريب النموذج
history = model.fit(X_train, y_train, epochs=25, batch_size=10, validation_split=0.2)

# تقييم النموذج
model.evaluate(X_test, y_test)

# بدء تجربة MLflow
with mlflow.start_run():
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
