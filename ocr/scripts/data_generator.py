import cv2
import git
import json
import matplotlib.pyplot as plt
import os

from PIL import ImageFont, ImageDraw, Image
from tools import load_from_json, load_training_data, MultiDimensionalArrayEncoder, METADATA_FOLDER, TRAIN_FOLDER


def generate_ocr_data(font_file_path, image_path, save_folder):
    count = 0
    spacing = 10
    font_sizes = [28]
    texts = [
        "((NEWHAVEN DISPLAY))\n4x20 CHARACTER OLEDS\nSLIM DESIGN ONLY 5MM\nOLED COLOR: <WHITE>",
        "ABCDEFGHIJKLM\nNOPQRSTUVWXYZ\n0123456789"
    ]
    # fill: font
    fill = ["white"]
    for font_size in font_sizes:
        for color in fill:
            for text in texts:
                image = Image.open(image_path)
                draw = ImageDraw.Draw(image)
                x = 15
                y = 15
                font = ImageFont.truetype(font_file_path, font_size)
                draw.text(
                    (x, y), text,
                    fill=color, font=font, spacing=spacing
                )  # (x, y) is the top left corner of the text to be drawn
                image.show()

                file_path = os.path.join(save_folder, f"train_{count}.jpg")
                image.save(file_path, "JPEG")
                count += 1


def generate_data(font_file_path, image_path, save_folder):
    image = Image.open(image_path)
    count = 0
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    chars = [char for char in chars]
    # symbols = "\'\"\\!@#$%^&*()-_=+,./<>?;:|~`[]{}"
    # symbols = [sym for sym in symbols]
    font_size = 20
    color = "white"
    spacing = 15
    font = ImageFont.truetype(font_file_path, font_size)
    x_values = list(range(5, image.size[0]-font_size, spacing))
    y_values = list(range(5, image.size[1]-font_size, spacing))
    for char in chars:
        for x in x_values:
            for y in y_values:
                image = Image.open(image_path)
                location = (x, y)
                draw = ImageDraw.Draw(image)
                file_path = os.path.join(save_folder, f"{char}_{count}.jpg")
                draw.text(
                    location, char,
                    fill=color, font=font, spacing=spacing
                )  # (x, y) is the top left corner of the text to be drawn
                # image.show()

                image.save(file_path, "JPEG")
                count += 1
        count = 0
    # index = 0
    # for sym in symbols:
    #     for x in x_values:
    #         for y in y_values:
    #             image = Image.open(image_path)
    #             location = (x, y)
    #             draw = ImageDraw.Draw(image)
    #             file_path = os.path.join(save_folder, f"symbol_{index}_{count}.jpg")
    #             draw.text(
    #                 location, sym,
    #                 fill=color, font=font, spacing=spacing
    #             )  # (x, y) is the top left corner of the text to be drawn
    #             # image.show()
    #
    #             image.save(file_path, "JPEG")
    #             count += 1
    #     count = 0
    #     index += 1


def save_to_json():
    file_name = "pixel_operator_dataset.json"
    file_path = os.path.join(METADATA_FOLDER, file_name)
    enc = MultiDimensionalArrayEncoder()
    pixel_dataset = load_training_data()
    json_string = enc.encode(pixel_dataset)
    with open(file_path, "w") as f:
        json.dump(json_string, f)


if __name__ == "__main__":
    cwd = os.getcwd()
    _git_repo = git.Repo(cwd, search_parent_directories=True)
    _git_root = _git_repo.git.rev_parse("--show-toplevel")
    _data_folder = os.path.join(
        _git_root, "data"
    )
    pixel_font_path = os.path.join(_data_folder, "fonts", "pixel_operator", "PixelOperatorMono.ttf")
    base_image = os.path.join(_data_folder, "images", "black_rectangle.jpg")
    # generate_data(pixel_font_path, base_image, TRAIN_FOLDER)
    save_to_json()
    # dataset = load_from_json()
