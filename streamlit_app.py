import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Настройки страницы
st.set_page_config(page_title="Прогноз отмены бронирования", layout="centered")

# Заголовок приложения
st.title("🏨 Прогнозирование отмены бронирования отеля")
st.write("Введите параметры, и модель предскажет вероятность отмены бронирования.")

# Загружаем данные
url = "https://raw.githubusercontent.com/AlexxxAI/ML_Project_1/refs/heads/master/Hotel_Reservations.csv"
df = pd.read_csv(url)

df = df.drop(columns=['Booking_ID'])

# Кодируем целевую переменную
if 'booking_status' in df.columns:
    df['booking_status'] = df['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})

# Кодируем категориальные признаки
label_encoder = LabelEncoder()
categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
for col in categorical_features:
    df[col] = label_encoder.fit_transform(df[col])

# Разделяем данные на признаки и целевую переменную
X = df.drop('booking_status', axis=1)
y = df['booking_status']

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Масштабируем числовые признаки
scaler = StandardScaler()
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Обучение модели Random Forest
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=10,
                                  min_samples_leaf=1, n_estimators=300, max_features='sqrt')
rf_model.fit(X_train, y_train)

# Интерфейс боковой панели с русскими названиями признаков
st.sidebar.header("Введите параметры бронирования:")
feature_names = {
    "no_of_adults": "Количество взрослых",
    "no_of_children": "Количество детей",
    "no_of_weekend_nights": "Число ночей (выходные)",
    "no_of_week_nights": "Число ночей (будни)",
    "required_car_parking_space": "Требуется парковка",
    "lead_time": "Дней до заезда",
    "arrival_month": "Месяц заезда",
    "arrival_date": "Дата заезда",
    "repeated_guest": "Повторный гость",
    "no_of_previous_cancellations": "Предыдущие отмены",
    "no_of_previous_bookings_not_canceled": "Ранее не отмененные бронирования",
    "avg_price_per_room": "Средняя цена за номер",
    "no_of_special_requests": "Число особых запросов"
}

user_input = {}
for col in X.columns:
    user_input[col] = st.sidebar.slider(feature_names.get(col, col), float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# Кнопка предсказания
if st.sidebar.button("🔍 Сделать предсказание"):
    prediction = rf_model.predict(input_scaled)
    prediction_proba = rf_model.predict_proba(input_scaled)
    
    st.subheader("📌 Результат предсказания:")
    if prediction[0] == 1:
        st.error("❌ Высокая вероятность отмены бронирования!")
    else:
        st.success("✅ Бронирование, скорее всего, не будет отменено.")
    
    st.write("**Вероятности предсказания:**")
    st.write(f"Не отменено: {prediction_proba[0][0]:.2f}, Отменено: {prediction_proba[0][1]:.2f}")

    # График важности признаков
    st.subheader("🔎 Важность признаков")
    feature_importances = pd.DataFrame({'Признак': X.columns, 'Важность': rf_model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Важность', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_importances['Признак'], feature_importances['Важность'], color='royalblue')
    ax.set_xlabel("Важность")
    ax.set_ylabel("Признак")
    ax.set_title("Важность признаков в модели Random Forest")
    st.pyplot(fig)
