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
    
    def detect_edges(self, image):
        """
        Выделяет контуры объектов в изображении
        
        Args:
            image: PIL Image объект
            
        Returns:
            PIL.Image: Изображение с выделенными контурами
        """
        # Конвертируем в градации серого
        grayscale = image.convert('L')
        
        # Применяем фильтры для выделения контуров
        edges = grayscale.filter(ImageFilter.FIND_EDGES)
        
        # Усиливаем контуры
        enhancer = ImageEnhance.Contrast(edges)
        edges = enhancer.enhance(2.0)
        
        return edges
    
    def create_depth_from_structure(self, image):
        """
        Создает карту глубины на основе структуры изображения
        
        Args:
            image: PIL Image объект
            
        Returns:
            numpy.ndarray: Улучшенная карта глубины
        """
        # Получаем контуры
        edges = self.detect_edges(image)
        edges_array = np.array(edges)
        
        # Конвертируем в градации серого
        grayscale = image.convert('L')
        grayscale_array = np.array(grayscale)
        
        # Создаем базовую карту глубины
        depth_map = 255 - grayscale_array
        
        # Усиливаем области с контурами
        edge_mask = edges_array > 50
        depth_map[edge_mask] = np.maximum(depth_map[edge_mask], 200)
        
        return depth_map
    
    def apply_morphological_operations(self, depth_map):
        """
        Применяет морфологические операции для улучшения формы
        
        Args:
            depth_map: numpy массив с картой глубины
            
        Returns:
            numpy.ndarray: Обработанная карта глубины
        """
        from scipy import ndimage
        
        # Создаем структурирующий элемент
        kernel = np.ones((3, 3))
        
        # Применяем морфологическое закрытие для заполнения пробелов
        depth_image = Image.fromarray(depth_map)
        
        # Применяем размытие для сглаживания
        blurred = depth_image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Увеличиваем контраст
        enhancer = ImageEnhance.Contrast(blurred)
        enhanced = enhancer.enhance(1.3)
        
        return np.array(enhanced)
    
    def image_to_depth_map(self, image):
        """
        Преобразует изображение в оптимизированную карту глубины для SIRDS
        
        Args:
            image: PIL Image объект
            
        Returns:
            numpy.ndarray: Карта глубины (значения от 0 до 255)
        """
        # Создаем карту глубины на основе структуры
        depth_map = self.create_depth_from_structure(image)
        
        # Применяем морфологические операции
        depth_map = self.apply_morphological_operations(depth_map)
        
        # Нормализуем значения
        depth_map = depth_map.astype(np.float32)
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min) * 255
        
        return depth_map.astype(np.uint8)
    
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
    
    def optimize_depth_for_stereogram(self, depth_map, intensity=1.0):
        """
        Оптимизирует карту глубины специально для стереограмм
        
        Args:
            depth_map: numpy массив с картой глубины
            intensity: Интенсивность эффекта
            
        Returns:
            numpy.ndarray: Оптимизированная карта глубины
        """
        # Применяем сигмоидальную функцию для улучшения контраста
        normalized = depth_map / 255.0
        
        # Сигмоидальное преобразование для усиления средних тонов
        sigmoid = 1 / (1 + np.exp(-10 * (normalized - 0.5)))
        
        # Применяем градиентное сглаживание для плавных переходов
        from scipy import ndimage
        gradient_x = ndimage.sobel(sigmoid, axis=1)
        gradient_y = ndimage.sobel(sigmoid, axis=0)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Уменьшаем резкие градиенты для лучшего восприятия
        smooth_factor = 1 - np.tanh(gradient_magnitude * 5)
        smoothed = sigmoid * smooth_factor + ndimage.gaussian_filter(sigmoid, sigma=1) * (1 - smooth_factor)
        
        # Возвращаем в диапазон 0-255
        return (smoothed * 255 * intensity).astype(np.uint8)

    def generate_sirds(self, input_image, dot_size=2, depth_intensity=1.0, 
                       pattern_width=100, output_width=800):
        """
        Генерирует SIRDS стереограмму из входного изображения с оптимизированным распознаванием формы
        
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
        
        # Создаем оптимизированную карту глубины для стереограмм
        depth_map = self.image_to_depth_map(resized_image)
        depth_map = self.optimize_depth_for_stereogram(depth_map, depth_intensity)
        
        # Создаем улучшенный паттерн на основе исходного изображения
        pattern = self.create_advanced_pattern(pattern_width, output_height, resized_image, dot_size)
        pattern_array = np.array(pattern)
        
        # Создаем выходное изображение
        sirds = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Заполняем начальный паттерн
        sirds[:, :pattern_width] = pattern_array
        
        # Константы для алгоритма SIRDS
        max_depth = 40  # Максимальное смещение в пикселях
        eye_separation = 64  # Расстояние между глазами в пикселях
        
        # Генерируем SIRDS с классическим алгоритмом
        for x in range(pattern_width, output_width):
            for y in range(output_height):
                # Получаем значение глубины
                depth_value = depth_map[y, x] / 255.0
                
                # Вычисляем смещение на основе глубины
                # Используем формулу стереографического смещения
                disparity = int(depth_value * max_depth)
                
                # Определяем источник пикселя
                source_x = x - pattern_width + disparity
                
                if source_x >= 0 and source_x < x:
                    sirds[y, x] = sirds[y, source_x]
                else:
                    # Используем базовый паттерн
                    pattern_x = x % pattern_width
                    sirds[y, x] = pattern_array[y, pattern_x]
        
        # Создаем финальное изображение без дополнительного размытия
        sirds_image = Image.fromarray(sirds)
        
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
