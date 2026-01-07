import cv2
import numpy as np


def main():
    vid = cv2.VideoCapture(0)

    if not vid.isOpened():
        print("не удалось открыть видео")
        return

    ascii_width = 60
    ascii_height = 50

    gradient = " .:!/r(lZ4H9W8$@"

    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            break

        # Уменьшаем до ASCII-размера
        frame_small = cv2.resize(
            frame, (ascii_width, ascii_height)
        )

        # Создаём чёрный фон
        out_h, out_w = frame.shape[:2]
        out_img = np.zeros(frame.shape, dtype=np.uint8)

        # Шаг сетки, чтобы равномерно разложить
        # ASCII_width * ASCII_height символов
        step_y = out_h / ascii_height
        step_x = out_w / ascii_width

        for y in range(ascii_height):  # строка ASCII
            for x in range(ascii_width):  # столбец ASCII
                b, g, r = frame_small[y, x]
                color = round((int(b) + int(g) + int(r)) / 3)
                ch = gradient[color // len(gradient)]

                # позиция вывода текста
                px = int(x * step_x)
                py = int((y + 1) * step_y)

                cv2.putText(
                    out_img,
                    ch,
                    (px, py),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.7,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("ASCII video", out_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
