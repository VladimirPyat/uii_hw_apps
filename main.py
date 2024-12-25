from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivymd.uix.button import MDRaisedButton
from kivymd.app import MDApp
from tkinter import filedialog, Tk
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
from kivymd.icon_definitions import md_icons
import tensorflow as tf
import numpy as np

# Словарь для перевода предсказаний на русский язык
translations = {
    "mosquito_net": "Москитная сетка",
    "rabbit": "Кролик",
    "cat": "Кошка",
    "dog": "Собака",
    "bird": "Птица",
    # Добавьте больше классов по необходимости
}


class ImageApp(MDApp):
    def build(self):
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        
        # Кнопки в горизонтальном ряду
        button_layout = BoxLayout(orientation='horizontal', size_hint=(2, 0.5), spacing=10)
        load_button = MDRaisedButton(
            text="Загрузить", size_hint=(None, None), width=150, height=50, md_bg_color=[0, 0.3, 0.4, 1], text_color=[1, 0, 0.5, 1]
        )
        load_button.bind(on_press=self.load_image)
        button_layout.add_widget(load_button)

        clear_button = MDRaisedButton(
            text="Очистить", size_hint=(None, None), width=150, height=50, text_color=[1, 0, 0.5, 1]
        )
        clear_button.bind(on_press=self.clear_image)
        button_layout.add_widget(clear_button)

        define_button = MDRaisedButton(
            text="Определить", size_hint=(None, None), width=150, height=50, text_color=[1, 0, 0.5, 1]
        )
        define_button.bind(on_press=self.define_image)
        button_layout.add_widget(define_button)

        layout.add_widget(button_layout)

        # Область для изображения
        self.image_view = Image(size_hint=(1, 3))
        layout.add_widget(self.image_view)


        # Область для текста результата
        self.result_label = Label(
            size_hint=(1, 1),
            text="Результат будет здесь",
            halign='center',
            valign='middle',
            color=[0, 0, 0, 1],
            font_size=20
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))  # Чтобы текст автоматически выравнивался
        layout.add_widget(self.result_label)
        self.result_label.font_name = 'Roboto'  # Установка шрифта
        self.result_label.font_size = 24  # Увеличение размера шрифта

        # Загрузка предобученной модели VGG19
        self.model = tf.keras.applications.VGG19(weights="imagenet")

        return layout

    def load_image(self, instance):
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])

        if file_path:
            self.image_path = file_path
            img = PILImage.open(file_path)
            img = img.resize((400, 400))
            self.image_view.texture = self.pil_to_texture(img)
            self.result_label.text = "Изображение загружено."

    def clear_image(self, instance):
        self.image_view.texture = None
        self.result_label.text = "Результат будет здесь"

    def define_image(self, instance):
        if hasattr(self, 'image_path'):
            img = PILImage.open(self.image_path).resize((224, 224))
            img_array = np.array(img)
            img_array = tf.keras.applications.vgg19.preprocess_input(img_array)  # Предобработка
            img_array = np.expand_dims(img_array, axis=0)

            predictions = self.model.predict(img_array)
            decoded_predictions = tf.keras.applications.vgg19.decode_predictions(predictions, top=1)

            # Получаем лучший результат и переводим на русский (если есть)
            _, class_name, score = decoded_predictions[0][0]
            translated_name = translations.get(class_name, class_name)
            self.result_label.text = f"На изображении: {translated_name} ({score:.2%})"
        else:
            self.result_label.text = "Сначала загрузите изображение."

    @staticmethod
    def pil_to_texture(pil_image):
        pil_image = pil_image.convert("RGBA")
        texture = Texture.create(size=pil_image.size)
        texture.blit_buffer(pil_image.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        texture.flip_vertical()
        return texture


if __name__ == '__main__':
    ImageApp().run()
