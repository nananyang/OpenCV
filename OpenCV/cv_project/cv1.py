import cv2


face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

mosaic_rate = 30

while True:
    ret, frame = cap.read()

    if not ret: break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)

    vis = frame.copy()

    for (x,y,w,h) in faces:
        print(x,y,w,h)


        roi_gray = frame[y:y+h, x:x+w]

        face_img = roi_gray
        face_img = cv2.resize(face_img, (w//mosaic_rate,h//mosaic_rate))
        face_img = cv2.resize(face_img, (w,h), interpolation=cv2.INTER_AREA)

        vis[y:y+h,x:x+w] = face_img


    cv2.imshow('frame', vis)

    if cv2.waitKey(20) >= 0: break

cap.release()
cv2.destroyAllWindows()

