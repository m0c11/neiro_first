import cv2
import numpy as np

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
        img = cv2.flip(
            img,
            1
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        
        
        # Находим лица
        faces = face_cascade.detectMultiScale(
            image= gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(50, 50)
        )
        
        # Список для хранения всех улыбок
        all_smiles = []
        
        # Находим улыбки во всем изображении
        smiles = smile_cascade.detectMultiScale(
            image= gray,
            scaleFactor=1.8,
            minNeighbors=25,
            minSize=(25, 25)
        )
        
        # Рисуем лица и проверяем улыбки
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(
                img= img, 
                pt1= (fx, fy), 
                pt2= (fx + fw, fy + fh), 
                color= (255, 0, 0), 
                thickness= 2
            )

            cv2.putText(
                img= img, 
                text= 'Face', 
                org= (fx, fy - 10),
                fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale= 0.5, 
                color= (255, 0, 0), 
                thickness= 1
            )
            
            # Проверяем каждую улыбку
            for (sx, sy, sw, sh) in smiles:
                smile_rect = (sx, sy, sw, sh)
                face_rect = (fx, fy, fw, fh)
                
                if is_smile_in_face(smile_rect, face_rect, threshold=0.6):
                    # Рисуем улыбку
                    cv2.rectangle(
                        img= img, 
                        pt1= (sx, sy), 
                        pt2= (sx+sw, sy+sh), 
                        color= (0, 255, 0),
                        thickness= 2
                    )
                    
                    cv2.putText(
                        img= img, 
                        text= 'Smile', 
                        org= (sx, sy-10),
                        fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale= 0.5, 
                        color= (0, 255, 0), 
                        thickness=1
                    )
                    
                    # Добавляем в список для статистики
                    all_smiles.append((sx, sy, sw, sh))
        
        # Статистика на кадре
        cv2.putText(
            img= img, 
            text= f'Faces: {len(faces)}', 
            org= (10, 30),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale= 0.7, 
            color= (255, 0, 0), 
            thickness= 2
        )
        
        cv2.putText(
            img= img, 
            text= f'Smiles in face: {len(all_smiles)}', 
            org= (10, 60),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale= 0.7, 
            color= (0, 255, 0), 
            thickness= 2
        )
        
        cv2.imshow('RESULT', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()