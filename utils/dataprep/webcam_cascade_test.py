import cv2

# Load the cascade
cascade = cv2.CascadeClassifier('cascade_2.xml')  # replace with your cascade file

# To use a video file instead of a camera, pass the video file path instead of the camera index
cap = cv2.VideoCapture(0)

frame_skip = 1  # Process every 5th frame
frame_count = 0

while True:
    # Read the frame
    ret, img = cap.read()

    # Resize the frame
    img = cv2.resize(img, (320, 240))

    frame_count += 1

    if frame_count % frame_skip == 0:
        # Convert to grayscale for the cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the objects
        objects = cascade.detectMultiScale(gray, 1.2, 5)

        # Draw the rectangle around each object in the original (color) image
        for (x, y, w, h) in objects:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()