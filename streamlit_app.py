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
user_input = {}

feature_map = {
    "no_of_adults": "Количество взрослых",
    "no_of_children": "Количество детей",
    "no_of_weekend_nights": "Ночей в выходные",
    "no_of_week_nights": "Ночей в будни",
    "required_car_parking_space": "Нужна парковка",
    "lead_time": "Дней до заезда",
    "arrival_month": "Месяц заезда",
    "arrival_date": "Дата заезда",
    "repeated_guest": "Повторный гость",
    "no_of_previous_cancellations": "Число отмен",
    "no_of_previous_bookings_not_canceled": "Число успешных броней",
    "avg_price_per_room": "Средняя цена номера",
    "no_of_special_requests": "Число пожеланий"
}

for col in X.columns:
    if col in ["no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights", "no_of_special_requests"]:
        user_input[col] = st.sidebar.number_input(feature_map[col], min_value=0, max_value=int(df[col].max()), value=int(df[col].mean()), step=1)
    elif col in ["required_car_parking_space", "repeated_guest"]:
        user_input[col] = st.sidebar.radio(feature_map[col], [0, 1], format_func=lambda x: "Да" if x == 1 else "Нет")
    elif col in ["arrival_month", "arrival_date"]:
        user_input[col] = st.sidebar.selectbox(feature_map[col], sorted(df[col].unique()))
    else:
        user_input[col] = st.sidebar.number_input(feature_map[col], float(df[col].min()), float(df[col].max()), float(df[col].mean()))

# Преобразуем ввод в DataFrame
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# Кнопка предсказания
if st.sidebar.button("🔍 Сделать предсказание"):
    prediction = rf_model.predict(input_scaled)
    prediction_proba = rf_model.predict_proba(input_scaled)
    result = "✅ Бронирование подтверждено" if prediction[0] == 0 else "⚠️ Высокая вероятность отмены!"
    st.subheader("Результат предсказания:")
    st.markdown(f"**{result}**")

    st.subheader("Вероятности предсказания:")
    st.write(f"- Подтверждение: {prediction_proba[0][0]:.2f}")
    st.write(f"- Отмена: {prediction_proba[0][1]:.2f}")

# График важности признаков
st.subheader("📊 Важность признаков")
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots()
importances.plot(kind='bar', ax=ax)
ax.set_title("Важность признаков")
ax.set_ylabel("Значимость")
st.pyplot(fig)
