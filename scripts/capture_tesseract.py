import cv2
import pytesseract
import os

from image_processing.image_analyzer_tesseract import ImageAnalyzer
from image_processing.image_capture import ImageCapture

# Add path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\zhoux\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


def extract_text_example():
    cwd = os.getcwd()
    save_path = os.path.join(
        cwd, "..", "data", "capture", "screenshot"
    )
    image_capture = ImageCapture(1)
    analyzer = ImageAnalyzer(image_capture, save_path)
    test_rpi_led_paths = [
        os.path.join(
            cwd, "..", "data", "test_images", "rpi_led.jpg"
        ),
        os.path.join(
            cwd, "..", "data", "test_images", "rpi_led_2.jpg"
        ),
        os.path.join(
            cwd, "..", "data", "test_images", "rpi_led_3.jpg"
        )
    ]

    save_paths = [
        os.path.join(
            cwd, "..", "data", "test_images", "rpi_led_cropped.jpg"
        ),
        os.path.join(
            cwd, "..", "data", "test_images", "rpi_led_2_cropped.jpg"
        ),
        os.path.join(
            cwd, "..", "data", "test_images", "rpi_led_3_cropped.jpg"
        )
    ]

    for i in range(len(test_rpi_led_paths)):
        img_path = test_rpi_led_paths[i]
        save_path = save_paths[i]
        analyzer.crop_image(img_path, save_path=save_path)

        print(analyzer.extract_text(save_path))
        analyzer.display_image(img_path)
        analyzer.display_image(save_path)


if __name__ == "__main__":
    cwd = os.getcwd()
    save_path = os.path.join(
        cwd, "..", "data", "capture", "screenshot"
    )
    # image_capture = ImageCapture(0)
    # analyzer = ImageAnalyzer(image_capture, save_path, lower_threshold=200, upper_threshold=255)
    # analyzer.run(show_edgemap=False, screenshot_on_pause=True)
    extract_text_example()
