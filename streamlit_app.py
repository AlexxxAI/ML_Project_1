import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Настройки страницы
st.set_page_config(page_title="Прогноз отмены бронирования", layout="centered")

# Заголовок приложения
st.title("🏨 Прогнозирование отмены бронирования отеля")
st.write("Введите параметры бронирования, и модель предскажет вероятность отмены.")

# Загрузка данных
df = pd.read_csv("https://raw.githubusercontent.com/AlexxxAI/ML_Project_1/refs/heads/master/Hotel_Reservations.csv")
df.drop(columns=['Booking_ID'], inplace=True)

# Кодирование категориальных признаков
label_encoders = {}
for col in ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Разделение данных
X = df.drop('booking_status', axis=1)
y = df['booking_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Масштабирование числовых данных
scaler = StandardScaler()
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Обучение модели
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=10, 
                                  min_samples_leaf=1, n_estimators=300, max_features='sqrt')
rf_model.fit(X_train, y_train)

# Интерфейс боковой панели
st.sidebar.header("Введите данные бронирования")

user_input = {}
feature_labels = {
    "lead_time": "Время до заезда (дней)",
    "no_of_adults": "Количество взрослых",
    "no_of_children": "Количество детей",
    "no_of_weekend_nights": "Ночи в выходные",
    "no_of_week_nights": "Ночи в будни",
    "required_car_parking_space": "Нужна парковка (0 - Нет, 1 - Да)",
    "no_of_special_requests": "Число особых запросов",
    "avg_price_per_room": "Средняя цена номера",
}

# Ввод параметров пользователем
for col, label in feature_labels.items():
    if col in ["no_of_adults", "no_of_children", "no_of_weekend_nights", "no_of_week_nights", "no_of_special_requests"]:
        user_input[col] = st.sidebar.number_input(label, min_value=0, max_value=int(df[col].max()), value=int(df[col].mean()))
    elif col == "required_car_parking_space":
        user_input[col] = st.sidebar.selectbox(label, [0, 1], format_func=lambda x: "Да" if x == 1 else "Нет")
    else:
        user_input[col] = st.sidebar.slider(label, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

# Подготовка данных пользователя
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# Предсказание
if st.sidebar.button("Сделать предсказание"):
    prediction = rf_model.predict(input_scaled)
    prediction_proba = rf_model.predict_proba(input_scaled)
    df_prediction_proba = pd.DataFrame(prediction_proba, columns=["Не отменено", "Отменено"])
    df_prediction_proba = df_prediction_proba.round(2)
    
    st.subheader("🔍 Результаты предсказания")
    
    if prediction[0] == 1:
        st.markdown("<div style='background-color: #ffcccc; padding: 10px; border-radius: 5px;'><strong>⚠️ Высокая вероятность отмены бронирования!</strong></div>", unsafe_allow_html=True)
    else:
        st.success("✅ Низкая вероятность отмены бронирования.")
    
    # Вывод вероятностей
    st.dataframe(
        df_prediction_proba,
        column_config={
            "Не отменено": st.column_config.ProgressColumn(
                "Не отменено",
                format="%.2f",
                width="medium",
                min_value=0,
                max_value=1
            ),
            "Отменено": st.column_config.ProgressColumn(
                "Отменено",
                format="%.2f",
                width="medium",
                min_value=0,
                max_value=1
            ),
        },
        hide_index=True
    )
    
    # Скачивание предсказания
    csv = df_prediction_proba.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Скачать предсказание", data=csv, file_name="prediction.csv", mime="text/csv")

# Визуализация важности признаков
st.subheader("📊 Важность признаков")
feature_importances = pd.DataFrame({
    'Признак': X.columns,
    'Значимость': rf_model.feature_importances_
}).sort_values(by='Значимость', ascending=False)
fig_importance = px.bar(feature_importances, x='Признак', y='Значимость', title='Важность признаков модели')
st.plotly_chart(fig_importance)
