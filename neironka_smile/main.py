import cv2
import numpy as np

def draw_smiley_face(img, x, y, size=50, is_smiling=True):
    """
    Рисует смайлик в указанных координатах
    """
    center = (x + size//2, y + size//2)
    
    # Цвет лица зависит от наличия улыбки
    face_color = (0, 255, 255) if is_smiling else (100, 100, 100)  # Желтый или серый
    
    # Лицо
    cv2.circle(img, center, size//2, face_color, -1)
    cv2.circle(img, center, size//2, (0, 0, 0), 2)  # Черная обводка
    
    # Глаза
    eye_radius = size//10
    left_eye = (center[0] - size//5, center[1] - size//10)
    right_eye = (center[0] + size//5, center[1] - size//10)
    
    cv2.circle(img, left_eye, eye_radius, (0, 0, 0), -1)
    cv2.circle(img, right_eye, eye_radius, (0, 0, 0), -1)
    
    # Рот - улыбка или прямая линия
    if is_smiling:
        # Улыбка (дуга)
        cv2.ellipse(img, center, (size//4, size//6), 0, 0, 180, (0, 0, 0), 2)
    else:
        # Прямой рот
        cv2.line(img, 
                (center[0] - size//4, center[1] + size//10),
                (center[0] + size//4, center[1] + size//10),
                (0, 0, 0), 2)

def is_smile_in_face(smile_rect, face_rect, threshold=0.8):
    sx, sy, sw, sh = smile_rect
    fx, fy, fw, fh = face_rect
    
    # Вычисляем площадь пересечения
    x_left = max(sx, fx)
    y_top = max(sy, fy)
    x_right = min(sx + sw, fx + fw)
    y_bottom = min(sy + sh, fy + fh)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    smile_area = sw * sh
    
    # Если достаточно большая часть улыбки находится внутри лица
    return intersection_area / smile_area >= threshold

def main():
    vid = cv2.VideoCapture(0)

    if not vid.isOpened():
        print("не удалось открыть видео")
        return
    
    smile_xml_path = '/home/w0rn/Документы/python/neiro/neironka_smile/neiro_ylibka.xml'
    face_xml_path = '/home/w0rn/Документы/python/neiro/neironka_smile/neiro_face.xml'
    
    face_cascade = cv2.CascadeClassifier(face_xml_path)
    smile_cascade = cv2.CascadeClassifier(smile_xml_path)
    
    while True:
        success, img = vid.read()
        if not success:
            break
        
        img = cv2.resize(img, (680, 540))
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Находим лица
        faces = face_cascade.detectMultiScale(
            image=gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        # Список для хранения всех улыбок
        all_smiles = []
        current_smile_state = False
        
        # Находим улыбки во всем изображении
        smiles = smile_cascade.detectMultiScale(
            image=gray,
            scaleFactor=1.8,
            minNeighbors=25,
            minSize=(25, 25)
        )
        
        # Рисуем лица и проверяем улыбки
        if len(faces) > 0:
            # Берем первое (главное) лицо
            fx, fy, fw, fh = faces[0]
            
            # Рисуем смайлик над лицом (на основном изображении)
            icon_x = fx + fw//2 - 25
            icon_y = fy - 80
            
            if icon_y >= 0 and icon_x >= 0 and icon_x + 50 < img.shape[1]:
                # Проверяем каждую улыбку
                for (sx, sy, sw, sh) in smiles:
                    smile_rect = (sx, sy, sw, sh)
                    face_rect = (fx, fy, fw, fh)
                    
                    if is_smile_in_face(smile_rect, face_rect, threshold=0.6):
                        
                        all_smiles.append((sx, sy, sw, sh))
                        current_smile_state = True
                
                # Рисуем интерактивный смайлик над лицом
                draw_smiley_face(img, icon_x, icon_y, 50, current_smile_state)
        
        # Показываем окно
        cv2.imshow('RESULT', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()