import cv2
import numpy as np
from pytesseract import pytesseract


class ImageAnalyzer:

    def __init__(
            self,
            image_capture,
            save_path,
            ext="jpg",
            lower_threshold=50,
            upper_threshold=255
    ):
        self._image_capture = image_capture
        self._image = None
        self._save_path = save_path
        self._ext = ext
        # Used for calculating the edge map
        self._lower_threshold = lower_threshold
        self._upper_threshold = upper_threshold

        self._frame_width = self._image_capture.video_dimensions[0]
        self._frame_height = self._image_capture.video_dimensions[1]
        max_box_width = self._frame_width*2 // 3
        max_box_height = self._frame_height*2 // 3
        min_box_width = self._frame_width // 3
        min_box_height = self._frame_height // 4

        self._max_bounds = [
            (self._frame_width // 2 - max_box_width // 2, self._frame_height // 2 - max_box_height // 2),  # top left
            (self._frame_width // 2 + max_box_width // 2, self._frame_height // 2 + max_box_height // 2),  # bottom right
        ]

        self._min_bounds = [
            (self._frame_width // 2 - min_box_width // 2, self._frame_height // 2 - min_box_height // 2),  # top left
            (self._frame_width // 2 + min_box_width // 2, self._frame_height // 2 + min_box_height // 2),  # bottom right
        ]
        self._min_area = min_box_width * min_box_height

    def run(self, show_edgemap=False, screenshot_on_pause=False):
        self._image_capture.start_recording()
        while True:
            image = self._image_capture.get_image()
            cropped = None
            key = cv2.waitKey(5)
            if key == ord(' ') or self._is_oled_on_screen(image):
                if screenshot_on_pause and self._is_oled_on_screen(image):
                    self.save_image(image)
                    cropped = self.crop_image(image)
                    text = self.extract_text(cropped)
                    if len(text) > 0:
                        print(f"Text detected:\n{self.extract_text(cropped)}")
                        self.save_image(cropped, f"{self._save_path}_cropped")
                cv2.waitKey()
            elif key == ord('q'):
                break
            # image is a numpy ndarray
            bounding_rect = self.get_largest_valid_bounding_rect(image)
            img_with_rect = self.draw_rectangle(image, bounding_rect, show_edgemap)
            img_with_rect = self._draw_bounds(image, show_edgemap)
            if cropped is not None:
                self.display_multiple_images([img_with_rect, cropped])
            else:
                cv2.imshow("frame", img_with_rect)
        # When everything is done, release the capture
        self._image_capture.stop_recording()
        cv2.destroyAllWindows()

    def crop_image(self, image, save_path=None):
        """'image' param must be an image, not a file path. Returns the cropped image."""
        if type(image) == str:
            image = cv2.imread(image)
        bounding_rect = self.get_largest_valid_bounding_rect(image)
        if bounding_rect is not None:
            x = bounding_rect[0]
            y = bounding_rect[1]
            w = bounding_rect[2]
            h = bounding_rect[3]
            cropped = image[y:y + h, x:x + w]
        else:
            print("No rectangles detected, not cropping image")
            cropped = image
        if save_path is not None:
            print(f"Saving cropped image to {save_path}_cropped.{self._ext}")
            cv2.imwrite(f"{save_path}_cropped.{self._ext}", cropped)
        return cropped

    def detect_rectangles(self, image):
        """Returns all rectangles found in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self._lower_threshold, self._upper_threshold)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, epsilon=0.01 * cv2.arcLength(contour, closed=True), closed=True)
            if len(approx) == 4:
                rectangles.append(contour)
        return rectangles

    def display_image(self, image):
        """Input can be an image or the path to an image. Displays the image until any key is pressed"""
        if type(image) == str:
            image = cv2.imread(image)
        cv2.imshow("frame", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_multiple_images(self, image_list):
        """image_list is a list of numpy arrays"""
        max_width = max([img.shape[1] for img in image_list])
        max_height = max([img.shape[0] for img in image_list])
        for i in range(len(image_list)):
            image = image_list[i]
            w = image.shape[1]
            h = image.shape[0]
            image_list[i] = np.pad(image, [
                (int(np.floor((max_height - h) / 2)), int(np.ceil((max_height - h) / 2))),
                (int(np.floor((max_width - w) / 2)), int(np.ceil((max_width - w) / 2))),
                (0, 0)
            ], mode="constant")
        image_array = np.hstack(image_list)
        cv2.imshow("frames", image_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_rectangle(self, image, bounding_rect, show_edgemap=False):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self._lower_threshold, self._upper_threshold)
        image_to_show = image
        if show_edgemap:
            image_to_show = edges
        if bounding_rect is not None:
            x = bounding_rect[0]
            y = bounding_rect[1]
            w = bounding_rect[2]
            h = bounding_rect[3]
            image_to_show = cv2.rectangle(image_to_show, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image_to_show

    def _draw_bounds(self, image, show_edgemap=False):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self._lower_threshold, self._upper_threshold)
        image_to_show = image
        if show_edgemap:
            image_to_show = edges
        image_to_show = cv2.rectangle(
            image_to_show,
            self._min_bounds[0],
            self._min_bounds[1],
            (0, 255, 0), 2
        )
        image_to_show = cv2.rectangle(
            image_to_show,
            self._max_bounds[0],
            self._max_bounds[1],
            (0, 255, 0), 2
        )
        return image_to_show

    def extract_text(self, image):
        """Input can be an image or the path to an image"""
        if type(image) == str:
            image = cv2.imread(image)
        text = pytesseract.image_to_string(image)
        return text

    def get_largest_valid_bounding_rect(self, image):
        rectangles = self.detect_rectangles(image)
        max_contour_area = 0
        max_index = -1
        for i, rectangle in enumerate(rectangles):
            if cv2.contourArea(rectangle) > max_contour_area:
                max_contour_area = cv2.contourArea(rectangle)
                max_index = i
        bounding_rect = None
        if len(rectangles) > 0 and cv2.contourArea(rectangles[max_index]) > self._min_area:
            bounding_rect = cv2.boundingRect(rectangles[max_index])  # (x, y, w, h)
        return bounding_rect

    def save_image(self, image, save_path=None):
        if save_path is None:
            save_path = self._save_path
        print(f"Saving image to {save_path}.{self._ext}")
        self.detect_rectangles(image)
        bounding_rect = self.get_largest_valid_bounding_rect(image)
        image_with_rectangle = self.draw_rectangle(image, bounding_rect, show_edgemap=False)
        cv2.imwrite(f"{save_path}.{self._ext}", image_with_rectangle)

    def _is_oled_on_screen(self, image):
        bounding_rect = self.get_largest_valid_bounding_rect(image)
        if bounding_rect is not None:
            return self._is_rectangle_in_bounds(bounding_rect)
        return False

    def _is_rectangle_in_bounds(self, rectangle):
        # rectangle = (x, y, w, h) where (x, y) = top left vertex
        return (
                self._max_bounds[0][0] <= rectangle[0] <= self._min_bounds[0][0] and
                self._max_bounds[0][1] <= rectangle[1] <= self._min_bounds[0][1] and
                self._min_bounds[1][0] <= rectangle[0] + rectangle[2] <= self._max_bounds[1][0] and
                self._min_bounds[1][1] <= rectangle[1] + rectangle[3] <= self._max_bounds[1][1]
        )
