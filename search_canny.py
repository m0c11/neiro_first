import cv2
import numpy as np

img = cv2.imread('inspector.jpg')

# создание пустого окна, по размерам картинки
picture = np.zeros(img.shape, dtype='uint8')

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (3, 3), 0)

img = cv2.Canny(img, 100, 140)

# con - список со всеми позициями контуров 
# hir - иерархия всех контуров, там будет порядок, что есть квадрат, а в этом квадрате линия
# функция находит контуры, сначала картинка или видео или что-то, режим получения контуров(ищет все именно этот режим), метод получения контуров
con, hir = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# рисуем картинку по контурам, сначала новое название, список всех контуров, я хуй знает что здесь, цвет, толщина обводки 
cv2.drawContours(picture, con, -1, (112, 203, 58), 1)

cv2.imshow('Result', picture)
cv2.waitKey(0) 