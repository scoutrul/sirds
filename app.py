import streamlit as st
import numpy as np
from PIL import Image
import io
from sirds_generator import SIRDSGenerator

# Настройка страницы
st.set_page_config(
    page_title="SIRDS Stereogram Generator",
    page_icon="👁️",
    layout="wide"
)

# Заголовок приложения
st.title("🔍 SIRDS Stereogram Generator")
st.markdown("**Генератор стереограмм из ваших изображений**")

# Информация о SIRDS
with st.expander("ℹ️ Что такое SIRDS и как их смотреть?"):
    st.markdown("""
    **SIRDS (Single Image Random Dot Stereograms)** - это особые изображения, содержащие скрытые 3D объекты.
    
    **Как смотреть стереограммы:**
    1. Поместите изображение на расстоянии 30-50 см от глаз
    2. Смотрите сквозь изображение, как будто пытаетесь увидеть что-то далеко за ним
    3. Расслабьте глаза и не фокусируйтесь на поверхности изображения
    4. Через некоторое время вы увидите скрытое 3D изображение
    
    **Совет:** Некоторым людям проще начать с приближения изображения к носу, а затем медленно отодвигать его.
    """)

# Боковая панель с настройками
st.sidebar.header("⚙️ Настройки генерации")

# Параметры SIRDS
dot_size = st.sidebar.slider(
    "Размер элементов паттерна",
    min_value=1,
    max_value=5,
    value=2,
    help="Размер элементов в фрактальном паттерне"
)

depth_intensity = st.sidebar.slider(
    "Интенсивность 3D эффекта",
    min_value=0.1,
    max_value=2.0,
    value=1.2,
    step=0.1,
    help="Насколько выражено трёхмерное восприятие объекта"
)

pattern_width = st.sidebar.slider(
    "Ширина паттерна",
    min_value=50,
    max_value=200,
    value=100,
    help="Ширина повторяющегося паттерна в пикселях"
)

output_width = st.sidebar.selectbox(
    "Ширина выходного изображения",
    [400, 600, 800, 1000, 1200],
    index=2,
    help="Ширина финальной стереограммы"
)

# Основная область
col1, col2 = st.columns(2)

with col1:
    st.header("📤 Загрузка изображения")
    
    uploaded_file = st.file_uploader(
        "Выберите изображение",
        type=['png', 'jpg', 'jpeg'],
        help="Поддерживаемые форматы: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Показываем превью загруженного изображения
        image = Image.open(uploaded_file)
        st.image(image, caption="Исходное изображение", use_container_width=True)
        
        # Информация об изображении
        st.info(f"Размер: {image.size[0]}x{image.size[1]} пикселей")

with col2:
    st.header("🎯 Результат")
    
    if uploaded_file is not None:
        # Загружаем изображение
        image = Image.open(uploaded_file)
        
        try:
            # Показываем индикатор загрузки
            with st.spinner("Генерируем стереограмму..."):
                # Создаем генератор SIRDS
                generator = SIRDSGenerator()
                
                # Генерируем стереограмму
                sirds_image = generator.generate_sirds(
                    image,
                    dot_size=dot_size,
                    depth_intensity=depth_intensity,
                    pattern_width=pattern_width,
                    output_width=output_width
                )
                
                # Показываем результат с возможностью открыть на весь экран
                st.image(sirds_image, caption="SIRDS Стереограмма", use_container_width=True)
                
                # Кнопка для просмотра в полноэкранном режиме
                if st.button("🔍 Открыть в полном размере", use_container_width=True):
                    st.image(sirds_image, caption="SIRDS Стереограмма - Полный размер")
                
                # Кнопка для скачивания
                buf = io.BytesIO()
                sirds_image.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Скачать стереограмму",
                    data=byte_im,
                    file_name="stereogram.png",
                    mime="image/png",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Ошибка при генерации стереограммы: {str(e)}")
            st.info("Попробуйте загрузить другое изображение или изменить настройки.")
    else:
        st.info("👆 Загрузите изображение для генерации стереограммы")

# Дополнительная информация
st.markdown("---")
st.markdown("### 💡 Советы для лучших результатов:")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Подходящие изображения:**
    - Простые формы и объекты
    - Высокий контраст
    - Четкие границы
    """)

with col2:
    st.markdown("""
    **Настройки:**
    - Начните с базовых настроек
    - Увеличьте интенсивность для более выраженного эффекта
    - Экспериментируйте с размером паттерна
    """)

with col3:
    st.markdown("""
    **Просмотр:**
    - Используйте хорошее освещение
    - Расслабьте глаза
    - Будьте терпеливы - может потребоваться время
    """)

# Футер
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Создано с ❤️ для любителей 3D искусства"
    "</div>",
    unsafe_allow_html=True
)
