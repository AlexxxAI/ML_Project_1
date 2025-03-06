import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Настройки страницы
st.set_page_config(page_title="Hotel Booking Cancellation Prediction", layout="centered")

st.title("🏨 Прогнозирование отмены бронирования номера отеля")
st.write("Введите параметры, и модель предскажет вероятность отмены бронирования.")

# Загрузка данных
url = "https://raw.githubusercontent.com/AlexxxAI/ML_Project_1/refs/heads/master/Hotel_Reservations.csv"
df = pd.read_csv(url)

# Удаляем колонку 'Booking_ID'
df = df.drop(columns=['Booking_ID'])

df['booking_status'] = df['booking_status'].replace({'Not_Canceled': 0, 'Canceled': 1})

# Кодируем категориальные признаки
label_encoder = LabelEncoder()
df['type_of_meal_plan'] = label_encoder.fit_transform(df['type_of_meal_plan'])
df['room_type_reserved'] = label_encoder.fit_transform(df['room_type_reserved'])
df['market_segment_type'] = label_encoder.fit_transform(df['market_segment_type'])

# Разделяем данные на признаки и целевую переменную
X = df.drop('booking_status', axis=1)
y = df['booking_status']

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Масштабируем числовые признаки
scaler = StandardScaler()
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Обучаем модель
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=10,
                                  min_samples_leaf=1, n_estimators=300, max_features='sqrt')
rf_model.fit(X_train, y_train)

# Сохраняем модель
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Интерфейс боковой панели
st.sidebar.header("Введите признаки:")
user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([user_input])

# Загружаем сохранённую модель и нормализатор
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
input_scaled = scaler.transform(input_df)

if st.sidebar.button("Сделать предсказание"):
    prediction = rf_model.predict(input_scaled)
    prediction_proba = rf_model.predict_proba(input_scaled)
    
    st.subheader("🔍 Результаты предсказания:")
    if prediction[0] == 1:
        st.error("⚠️ Высокая вероятность отмены бронирования!")
    else:
        st.success("✅ Бронирование с высокой вероятностью состоится.")
    
    df_prediction_proba = pd.DataFrame(prediction_proba, columns=["Не отменено", "Отменено"])
    st.dataframe(df_prediction_proba.round(2))
    
    csv = df_prediction_proba.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Скачать предсказание", data=csv, file_name="prediction.csv", mime="text/csv")

# Визуализация данных
st.subheader("📊 Визуализация данных")
top_features = X_train.corrwith(y_train).abs().sort_values(ascending=False).index[:2]
fig = px.scatter(df, x=top_features[0], y=top_features[1], color="booking_status", title="Два наиболее коррелирующих признака")
st.plotly_chart(fig)
