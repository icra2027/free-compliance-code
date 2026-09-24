import cv2

cap = cv2.VideoCapture(7)

ret, frame = cap.read()

while ret:
    cv2.imshow('Camera Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()