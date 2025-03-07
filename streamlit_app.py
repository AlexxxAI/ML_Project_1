import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Кэшируем загрузку данных
@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df.drop(columns=['Booking_ID'], inplace=True)
    return df

# Загружаем данные
url = "https://raw.githubusercontent.com/AlexxxAI/ML_Project_1/refs/heads/master/Hotel_Reservations.csv"
df = load_data(url)

# Кодируем категориальные признаки
label_encoders = {}
categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Разделяем данные на признаки и целевую переменную
X = df.drop(columns=['booking_status', 'arrival_year', 'arrival_date'], axis=1)
y = (df['booking_status'] == 'Canceled').astype(int)

# Интерфейс Streamlit
st.title("🔮 Прогноз отмены бронирования")
st.sidebar.header("Введите данные о бронировании")

# Отображаем датасет
with st.expander("Data"):
    st.write(df)

# Разделяем на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Масштабируем числовые признаки
scaler = StandardScaler()
numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

# Обучаем модель
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=10,
                                   min_samples_leaf=1, n_estimators=300, max_features='sqrt')
    model.fit(X_train, y_train)
    return model
rf_model = train_model(X_train, y_train)
y_pred = rf_model.predict(X_test)

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
user_input['arrival_month'] = st.sidebar.slider("Месяц заезда", 1, 12, 6)
user_input['market_segment_type'] = st.sidebar.selectbox("Тип клиента", label_encoders['market_segment_type'].classes_)
user_input['repeated_guest'] = st.sidebar.checkbox("Постоянный клиент?")
user_input['no_of_previous_cancellations'] = st.sidebar.slider("Предыдущие отмены", 0, 10, 0)
user_input['no_of_previous_bookings_not_canceled'] = st.sidebar.slider("Бронирований без отмен", 0, 50, 0)
user_input['avg_price_per_room'] = st.sidebar.slider("Средняя цена номера", 0, 500, 100)
user_input['no_of_special_requests'] = st.sidebar.slider("Количество специальных запросов (например, доп. кровать)", 0, 5, 0)

# Преобразуем ввод пользователя в DataFrame
input_df = pd.DataFrame([user_input])

with st.expander('Input features'):
    st.write('**Input booking**')
    st.dataframe(input_df)
    st.write('**Combined bookings data** (input row + original data)')
    combined_df = pd.concat([input_df, X], axis=0)
    st.dataframe(combined_df)

# Кодируем категориальные признаки
for col in categorical_features:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Масштабируем числовые признаки
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

with st.expander('Data preparation'):
    st.write('**Encoded X (input booking)**')
    st.dataframe(input_df)
    st.write('**Encoded y**')
    st.write(y)

# Кнопка предсказания
if st.sidebar.button("🔍 Сделать предсказание"):
    prediction = rf_model.predict(input_df)[0]
    prediction_proba = rf_model.predict_proba(input_df)[0]
    
    st.subheader("📊 Результат предсказания:")
    if prediction == 1:
        st.error("⚠️ Высокая вероятность отмены бронирования!")
    else:
        st.success("✅ Бронирование скорее всего не будет отменено.")
    
    st.progress(int(prediction_proba[1] * 100))
    st.write(f"**Вероятность отмены:** {prediction_proba[1]:.2f}")

    # Создаем DataFrame для визуализации прогресс-баров
    df_prediction_proba = pd.DataFrame({
        'Canceled': [prediction_proba[1]],
        'Not Canceled': [prediction_proba[0]]
    })

    # Отображаем вероятности с прогресс-барами
    st.subheader('📊 Вероятности предсказания')
    st.dataframe(
        df_prediction_proba,
        column_config={
            'Canceled': st.column_config.ProgressColumn(
                'Canceled',
                format='%.1f',
                width='medium',
                min_value=0,
                max_value=1
            ),
            'Not Canceled': st.column_config.ProgressColumn(
                'Not Canceled',
                format='%.1f',
                width='medium',
                min_value=0,
                max_value=1
            ),
        },
        hide_index=True
    )

    # Визуализация важности признаков
    st.subheader("📈 Data Visualization")

    # Визуализация важности признаков
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_1 = px.bar(feature_importances, title="Feature Importance", labels={'value': 'Важность', 'index': 'Признаки'})
    st.plotly_chart(fig_1)

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Не отменена", "Отменена"]
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        labels={'x': 'Предсказано', 'y': 'Истинное значение'})
    fig_cm.update_xaxes(tickvals=[0, 1], ticktext=labels)
    fig_cm.update_yaxes(tickvals=[0, 1], ticktext=labels)
    fig_cm.update_layout(title_text="Матрица ошибок")
    st.plotly_chart(fig_cm)

    # Распределение отмененных бронирований по времени до заезда
    df['booking_status'] = df['booking_status'].apply(lambda x: 'Canceled' if x == 'Canceled' else 'Not Canceled')
    fig_3 = px.histogram(df, x='lead_time', color='booking_status', barmode='group',
                         title='Распределение отмененных бронирований по времени до заезда',
                         color_discrete_sequence=['#FF0000', '#0000FF'])
    fig_3.update_layout(
        width=1000,
        height=800
    )
    st.plotly_chart(fig_3)

    # Влияние количества специальных запросов на отмену бронирования (no_of_special_requests)
    fig_4 = px.bar(df.groupby(['no_of_special_requests', 'booking_status']).size().reset_index(name='count'),
                   x='no_of_special_requests', y='count', color='booking_status', barmode='group',
                   title='Влияние количества специальных запросов на отмену бронирования')
    st.plotly_chart(fig_4)

    # Влияние средней цены на номер на отмену бронирования (avg_price_per_room)
    fig_5 = px.scatter(df, x='avg_price_per_room', y='lead_time', color='booking_status',
                       title='Зависимость отмены бронирования от средней цены номера')
    st.plotly_chart(fig_5)

    # Распределение отмененных бронирований по типу клиента (market_segment_type)
    df['market_segment_type'] = df['market_segment_type'].astype(str)
    fig_6 = px.bar(df.groupby(['market_segment_type', 'booking_status']).size().reset_index(name='count'),
                   x='market_segment_type', y='count', color='booking_status', barmode='group',
                   title='Распределение отмененных бронирований по типу клиента',
                   category_orders={"market_segment_type": ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary']})
    st.plotly_chart(fig_6)

    # График 5: Влияние месяца заезда на отмену бронирования (arrival_month)
    fig_7 = px.bar(df.groupby(['arrival_month', 'booking_status']).size().reset_index(name='count'),
                   x='arrival_month', y='count', color='booking_status', barmode='group',
                   title='Влияние месяца заезда на отмену бронирования')
    st.plotly_chart(fig_7)
