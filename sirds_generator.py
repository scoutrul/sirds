import numpy as np
from PIL import Image, ImageDraw
import random

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
    
    def create_random_pattern(self, width, height, dot_size=2):
        """
        Создает случайный паттерн точек
        
        Args:
            width: Ширина паттерна
            height: Высота паттерна
            dot_size: Размер точек
            
        Returns:
            PIL.Image: Изображение с случайным паттерном
        """
        # Создаем изображение
        pattern = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(pattern)
        
        # Генерируем случайные точки
        num_dots = (width * height) // (dot_size * dot_size * 4)
        
        for _ in range(num_dots):
            x = random.randint(0, width - dot_size)
            y = random.randint(0, height - dot_size)
            
            # Случайный цвет точки
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            
            # Рисуем точку
            draw.ellipse(
                [x, y, x + dot_size, y + dot_size],
                fill=color
            )
        
        return pattern
    
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
        
        # Создаем карту глубины
        depth_map = self.image_to_depth_map(resized_image)
        
        # Создаем случайный паттерн
        pattern = self.create_random_pattern(pattern_width, output_height, dot_size)
        pattern_array = np.array(pattern)
        
        # Создаем выходное изображение
        sirds = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Заполняем начальный паттерн
        sirds[:, :pattern_width] = pattern_array
        
        # Генерируем SIRDS
        for x in range(pattern_width, output_width):
            for y in range(output_height):
                # Получаем значение глубины (нормализованное)
                depth_value = depth_map[y, x] / 255.0
                
                # Вычисляем смещение на основе глубины
                # Чем больше глубина, тем больше смещение
                shift = int(depth_value * depth_intensity * 20)
                
                # Ограничиваем смещение
                shift = max(0, min(shift, pattern_width - 1))
                
                # Копируем пиксель со смещением
                source_x = x - pattern_width + shift
                if source_x >= 0:
                    sirds[y, x] = sirds[y, source_x]
                else:
                    # Если смещение выходит за границы, используем паттерн
                    sirds[y, x] = pattern_array[y, (x - shift) % pattern_width]
        
        # Преобразуем обратно в PIL Image
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
