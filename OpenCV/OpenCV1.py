import cv2
import numpy as np
import pickle

font=cv2.FONT_HERSHEY_SIMPLEX

def faceDetect():
    face_cascade= cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
    # 얼굴인식 회로?
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # 인식기 만들기
    recognizer.read('trainner.yml')
    # 인식기에 사용될 학습모델 불러오기

    labels = {'person_name': 1}
    # 라벨 딕셔너리 선언

    with open('lables.pickle', 'rb') as f: # labels.pickle 데이터를 바이트형식으로 읽기
        og_labels = pickle.load(f)
        # 읽은 파일을 불러와 ob_labels 에 담기
        labels = {v:k for k,v in og_labels.items()}
        # or_labels의 내용을 k,v 에 담고 서로내용을 바꾸어 v,k에 담기
    
    try:

        cap=cv2.VideoCapture(0)
        # 웹캠 활성화 코드

    except:
        print('카메라 로딩 실패')
        return
    j = 0
    while True: 
        ret,frame = cap.read()
        # 웹캠을 프레임단위로 읽습니다 제대로 읽으면 ret이 True 아니면 False을 나타내고 frame 은 읽은 프레임

        if not ret:
            return



        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 프레임을 흑백으로 전환
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
        # 흑백으로 전환된 프레임에서 얼굴을 찾는다

        for(x,y,w,h) in faces: # 얼굴의 좌표값 담기
            print(x,y,w,h)
            
            roi_gray = gray[y:y+h, x:x+w] # (ycord_start, ycord_end)
            # 얼굴 네모칸
            roi_color = frame[y-25:y+h+25, x-25:x+w+25]
            # 사진저장 할 얼굴칸 설정

            # recognize? deep learned model predict keras tensorflow pytorch scikit learn 
            id_, conf = recognizer.predict(roi_gray) # 인식기에 얼굴을 대입해 id_ = 학습모델값 conf = 정확도
            if conf>= 10 and conf <=99:
                print(id_)
                print(labels[id_])
                print(conf)

            #img_item = "%d.png"%j
            #cv2.imwrite('C:/Users/W/PycharmProjects/test/venv/OpenCV/images/yongdae/'+img_item, roi_color)

            print(j)
            j+=1

            color = (255,0,0) # BGR 0-255
            stroke = 2 # 두께
            name = labels[id_]
            end_cord_x = x+w
            end_cord_y = y+h
            cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke) # 네모그리기
            subitems = smile_cascade.detectMultiScale(roi_gray) # 표정 찾기
            for (ex,ey,ew,eh) in subitems:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            cv2.putText(frame,str(int(conf))+name,(x-5,y-5),font,0.9,(255,255,0),2,cv2.LINE_AA) # 텍스트 출력
    # 위코드는 얼굴을 인식하는 사격형에 대한 소스, 텍스틑 소스
    
        cv2.imshow('frame',frame) # 영상을 출력하는 소스
        
        if cv2.waitKey(20) & 0xFF == ord('q'): break # 20ms 입력대기 및 'q' 버튼 누를때 for문 종료
        #if cv2.waitKey(1)>0: break
            
    cap.release()
    # cap 객체 해제
    cv2.destroyAllWindows()
    # 모든 윈도우창 종료

faceDetect() # 함수 실행

