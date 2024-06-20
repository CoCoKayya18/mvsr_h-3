import cv2

# Replace this URL with your RTMP stream URL
rtmp_stream_url = 'rtmp://192.168.0.4/live/WSL_Stream'

cap = cv2.VideoCapture(rtmp_stream_url)

if not cap.isOpened():
    print("Error: Could not open RTMP stream.")
else:
    print("RTMP stream opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

