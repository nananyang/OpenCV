import cv2

song = cv2.VideoCapture("TheOcean.mp4")

while True:
    if(song.get(cv2.CAP_PROP_POS_FRAMES) == song.get(cv2.CAP_PROP_FRAME_COUNT)):
        song.open('TheOcean.mp4')
        
    ret, frame = song.read()
    cv2.imshow('VideoFrame', frame)
        
    if cv2.waitKey(30) > 0: break
            
song.release()
cv2.destroyAllWindows()