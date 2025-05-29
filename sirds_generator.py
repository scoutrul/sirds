import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import random
import math

class SIRDSGenerator:
    """
    Генератор SIRDS (Single Image Random Dot Stereograms) стереограмм
    """
    
    def __init__(self):
        """Инициализация генератора"""
        pass
    
    def image_to_depth_map(self, image):
        """
        Преобразует изображение в карту глубины
        
        Args:
            image: PIL Image объект
            
        Returns:
            numpy.ndarray: Карта глубины (значения от 0 до 255)
        """
        # Конвертируем в градации серого
        grayscale = image.convert('L')
        
        # Преобразуем в numpy массив
        depth_map = np.array(grayscale)
        
        # Инвертируем значения (темные области = близко, светлые = далеко)
        depth_map = 255 - depth_map
        
        return depth_map
    
    def extract_color_palette(self, image, num_colors=8):
        """
        Извлекает основные цвета из изображения
        
        Args:
            image: PIL Image объект
            num_colors: Количество цветов для извлечения
            
        Returns:
            list: Список основных цветов в формате RGB
        """
        # Уменьшаем изображение для ускорения обработки
        small_image = image.resize((100, 100))
        
        # Квантизируем цвета
        quantized = small_image.quantize(colors=num_colors)
        palette = quantized.getpalette()
        
        # Извлекаем уникальные цвета
        colors = []
        for i in range(num_colors):
            r = palette[i * 3]
            g = palette[i * 3 + 1]
            b = palette[i * 3 + 2]
            colors.append((r, g, b))
        
        return colors
    
    def create_fractal_pattern(self, width, height, colors, iterations=100):
        """
        Создает фрактальный паттерн с использованием цветов из исходного изображения
        
        Args:
            width: Ширина паттерна
            height: Высота паттерна
            colors: Список цветов для использования
            iterations: Количество итераций для фрактала
            
        Returns:
            PIL.Image: Изображение с фрактальным паттерном
        """
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Параметры фрактала Мандельброта
        xmin, xmax = -2.0, 1.0
        ymin, ymax = -1.5, 1.5
        
        for y in range(height):
            for x in range(width):
                # Преобразуем координаты пикселя в комплексную плоскость
                c = complex(xmin + (xmax - xmin) * x / width,
                          ymin + (ymax - ymin) * y / height)
                
                z = 0
                iteration_count = 0
                for iteration_count in range(iterations):
                    if abs(z) > 2:
                        break
                    z = z * z + c
                
                # Используем количество итераций для выбора цвета
                color_index = iteration_count % len(colors)
                pattern[y, x] = colors[color_index]
        
        return Image.fromarray(pattern)
    
    def create_noise_pattern(self, width, height, colors, noise_intensity=0.3):
        """
        Создает шумовой паттерн с цветами из исходного изображения
        
        Args:
            width: Ширина паттерна
            height: Высота паттерна
            colors: Список цветов для использования
            noise_intensity: Интенсивность шума
            
        Returns:
            PIL.Image: Изображение с шумовым паттерном
        """
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                # Создаем многослойный шум
                noise_value = 0
                frequency = 1.0
                amplitude = 1.0
                
                for _ in range(4):  # 4 октавы шума
                    noise_value += amplitude * math.sin(frequency * x * 0.01) * math.cos(frequency * y * 0.01)
                    frequency *= 2
                    amplitude *= 0.5
                
                # Нормализуем и добавляем случайность
                noise_value = (noise_value + 1) / 2  # Нормализация в [0, 1]
                noise_value += random.random() * noise_intensity
                
                # Выбираем цвет на основе шума
                color_index = int(noise_value * len(colors)) % len(colors)
                pattern[y, x] = colors[color_index]
        
        return Image.fromarray(pattern)
    
    def create_advanced_pattern(self, width, height, source_image, dot_size=2):
        """
        Создает улучшенный паттерн с использованием фракталов и цветов исходного изображения
        
        Args:
            width: Ширина паттерна
            height: Высота паттерна
            source_image: Исходное изображение для извлечения цветов
            dot_size: Размер элементов паттерна
            
        Returns:
            PIL.Image: Улучшенный паттерн
        """
        # Извлекаем основные цвета из исходного изображения
        colors = self.extract_color_palette(source_image, num_colors=12)
        
        # Создаем фрактальную основу
        fractal_pattern = self.create_fractal_pattern(width, height, colors[:6])
        
        # Добавляем шумовой слой
        noise_pattern = self.create_noise_pattern(width, height, colors[6:], noise_intensity=0.2)
        
        # Смешиваем паттерны
        fractal_array = np.array(fractal_pattern)
        noise_array = np.array(noise_pattern)
        
        # Комбинируем с весами
        combined = (fractal_array * 0.7 + noise_array * 0.3).astype(np.uint8)
        
        return Image.fromarray(combined)
    
    def generate_sirds(self, input_image, dot_size=2, depth_intensity=1.0, 
                       pattern_width=100, output_width=800):
        """
        Генерирует SIRDS стереограмму из входного изображения
        
        Args:
            input_image: PIL Image объект для преобразования
            dot_size: Размер точек в паттерне
            depth_intensity: Интенсивность 3D эффекта
            pattern_width: Ширина повторяющегося паттерна
            output_width: Ширина выходного изображения
            
        Returns:
            PIL.Image: SIRDS стереограмма
        """
        # Изменяем размер входного изображения
        aspect_ratio = input_image.height / input_image.width
        output_height = int(output_width * aspect_ratio)
        
        # Масштабируем входное изображение
        resized_image = input_image.resize((output_width, output_height), Image.Resampling.LANCZOS)
        
        # Создаем улучшенную карту глубины
        depth_map = self.image_to_depth_map(resized_image)
        depth_map = self.enhance_depth_map(depth_map, blur_radius=1)
        
        # Создаем улучшенный паттерн на основе исходного изображения
        pattern = self.create_advanced_pattern(pattern_width, output_height, resized_image, dot_size)
        pattern_array = np.array(pattern)
        
        # Создаем выходное изображение
        sirds = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Заполняем начальный паттерн
        sirds[:, :pattern_width] = pattern_array
        
        # Генерируем SIRDS с улучшенным алгоритмом
        for x in range(pattern_width, output_width):
            for y in range(output_height):
                # Получаем значение глубины (нормализованное)
                depth_value = depth_map[y, x] / 255.0
                
                # Улучшенное вычисление смещения для лучшего 3D эффекта
                base_shift = depth_value * depth_intensity * 30
                
                # Добавляем нелинейность для более выразительного эффекта
                nonlinear_factor = depth_value ** 1.5
                shift = int(base_shift * nonlinear_factor)
                
                # Ограничиваем смещение
                max_shift = min(pattern_width // 2, 40)
                shift = max(-max_shift, min(shift, max_shift))
                
                # Вычисляем исходную позицию с учетом смещения
                source_x = x - pattern_width - shift
                
                if source_x >= 0 and source_x < output_width:
                    sirds[y, x] = sirds[y, source_x]
                else:
                    # Используем паттерн с модификацией для лучшего качества
                    pattern_x = (x - shift) % pattern_width
                    sirds[y, x] = pattern_array[y, pattern_x]
        
        # Применяем легкое размытие для сглаживания артефактов
        sirds_image = Image.fromarray(sirds)
        sirds_image = sirds_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return sirds_image
    
    def enhance_depth_map(self, depth_map, blur_radius=2):
        """
        Улучшает карту глубины для лучшего 3D эффекта
        
        Args:
            depth_map: numpy массив с картой глубины
            blur_radius: Радиус размытия для сглаживания
            
        Returns:
            numpy.ndarray: Улучшенная карта глубины
        """
        # Преобразуем в PIL для применения фильтров
        depth_image = Image.fromarray(depth_map)
        
        # Применяем легкое размытие для сглаживания
        from PIL import ImageFilter
        blurred = depth_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Увеличиваем контраст
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(blurred)
        enhanced = enhancer.enhance(1.5)
        
        return np.array(enhanced)
    
    def create_text_depth_map(self, text, width, height, font_size=50):
        """
        Создает карту глубины из текста
        
        Args:
            text: Текст для создания карты глубины
            width: Ширина изображения
            height: Высота изображения
            font_size: Размер шрифта
            
        Returns:
            PIL.Image: Изображение с текстом как карта глубины
        """
        # Создаем изображение
        image = Image.new('L', (width, height), 0)  # Черный фон
        draw = ImageDraw.Draw(image)
        
        # Пытаемся найти системный шрифт
        try:
            from PIL import ImageFont
            # Пробуем загрузить системный шрифт
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            # Если не получается, используем стандартный
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
            except:
                font = None
        
        # Получаем размеры текста
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(text) * 10  # Примерная оценка
            text_height = 20
        
        # Центрируем текст
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Рисуем текст белым цветом
        draw.text((x, y), text, fill=255, font=font)
        
        return image
