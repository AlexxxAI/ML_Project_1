import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Загружаем данные
url = "https://raw.githubusercontent.com/AlexxxAI/ML_Project_1/refs/heads/master/Hotel_Reservations.csv"
df = pd.read_csv(url)
df.drop(columns=['Booking_ID'], inplace=True)

# Кодируем категориальные признаки
label_encoders = {}
categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Разделяем данные на признаки и целевую переменную
X = df.drop('booking_status', axis=1)
y = (df['booking_status'] == 'Canceled').astype(int)  # Преобразуем в 0 и 1

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

# Интерфейс Streamlit
st.title("🔮 Прогноз отмены бронирования")
st.sidebar.header("Введите данные о бронировании")

# Поля для ввода параметров
user_input = {}
user_input['no_of_adults'] = st.sidebar.slider("Количество взрослых", 1, 5, 2)
user_input['no_of_children'] = st.sidebar.slider("Количество детей", 0, 5, 0)
user_input['no_of_weekend_nights'] = st.sidebar.slider("Ночи в выходные", 0, 5, 1)
user_input['no_of_week_nights'] = st.sidebar.slider("Ночи в будни", 0, 10, 2)
user_input['type_of_meal_plan'] = st.sidebar.selectbox("Тип питания", label_encoders['type_of_meal_plan'].classes_)
user_input['required_car_parking_space'] = st.sidebar.checkbox("Нужна парковка?")
user_input['room_type_reserved'] = st.sidebar.selectbox("Тип номера", label_encoders['room_type_reserved'].classes_)
user_input['lead_time'] = st.sidebar.slider("Дни до заезда", 0, 500, 100)
user_input['arrival_year'] = st.sidebar.slider("Год заезда", 2017, 2021, 2019)
user_input['arrival_month'] = st.sidebar.slider("Месяц заезда", 1, 12, 6)
user_input['arrival_date'] = st.sidebar.slider("Дата заезда", 1, 31, 15)
user_input['market_segment_type'] = st.sidebar.selectbox("Тип клиента", label_encoders['market_segment_type'].classes_)
user_input['repeated_guest'] = st.sidebar.checkbox("Постоянный клиент?")
user_input['no_of_previous_cancellations'] = st.sidebar.slider("Предыдущие отмены", 0, 10, 0)
user_input['no_of_previous_bookings_not_canceled'] = st.sidebar.slider("Бронирований без отмен", 0, 50, 0)
user_input['avg_price_per_room'] = st.sidebar.slider("Средняя цена номера", 0, 500, 100)
user_input['no_of_special_requests'] = st.sidebar.slider("Спецзапросы", 0, 5, 0)

# Преобразуем ввод пользователя в DataFrame
input_df = pd.DataFrame([user_input])

# Кодируем категориальные признаки
for col in categorical_features:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Масштабируем числовые признаки
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# Инициализируем переменную состояния, если ее нет
if "predict" not in st.session_state:
    st.session_state.predict = False

# Обработчик нажатия кнопки
def make_prediction():
    st.session_state.predict = True

# Кнопка предсказания
if st.sidebar.button("🔍 Сделать предсказание", on_click=make_prediction):
    st.session_state.predict = True

# Выполняем предсказание только если кнопка нажата
if st.session_state.predict:
    prediction = rf_model.predict(input_df)[0]
    prediction_proba = rf_model.predict_proba(input_df)[0]
    
    st.subheader("📊 Результат предсказания:")
    if prediction == 1:
        st.error("⚠️ Высокая вероятность отмены бронирования!")
    else:
        st.success("✅ Бронирование скорее всего не будет отменено.")
    
    st.progress(int(prediction_proba[1] * 100))
    st.write(f"**Вероятность отмены:** {prediction_proba[1]:.2f}")

# Визуализация важности признаков
st.subheader("📈 Важность признаков")
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
fig = px.bar(feature_importances, title="Feature Importance", labels={'value': 'Важность', 'index': 'Признаки'})
st.plotly_chart(fig)
