import cv2
import pickle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
import winsound
import datetime
import time


# LOAD TRAINNED RECOGNISER
# =============================================================================
# =============================================================================


# Path to the dataset and trained recognizer
recognizer_path = 'trainer.yml'

# Load the trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(recognizer_path)

labels = {"name": 1}
with open("labels.pickle", "rb") as f:
    initial_labels = pickle.load(f)
    labels = {v: k for k, v in initial_labels.items()}

# Load the haarcascade classifiers for face detection
face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')


 #SETTING UP THE VIDEO CAPTURE AND THE WINSOUND MODULE
# =============================================================================
# =============================================================================
# Set up the video capture
cap = cv2.VideoCapture(0)
cap.set(3,640)#setting width of the capture frame
cap.set(4,480)# setting the height of the capture frame

 

#setting up video writter
current_time =datetime.datetime.now().strftime("%Y -%m-%d_%H-%M-%S")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'captured_video_{current_time}.avi',fourcc,20.0,(640,480))


# Set up the alert sound
alert_sound = lambda: winsound.PlaySound('alert-alarm.wav',winsound.SND_ASYNC)
allow_sound= lambda:  winsound.PlaySound('access-allowed.wav', winsound.SND_ALIAS)



#READING FROM FRAME
# =============================================================================
# =============================================================================


while True:
    # Read the video frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to initialise video capture")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)

    # Draw a white rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Recognize the face
        roi_gray = gray[y:y + h, x:x + w]
        person_identity, confidence = recognizer.predict(roi_gray)

        # Label the face with the name and access granted or denied
        if confidence < 80:
            # Get the name from the dataset folder

            cv2.putText(frame, labels[person_identity], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, 'Access Granted', (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #play winsound access granted sound
            allow_sound()

                       # =============================================================================

            # IF THE VISITOR IS NOT RECOGNISED
        else:
            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, 'Access Denied', (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # CAPTURE IMAGE
            cv2.imwrite(f'unknown{current_time}.jpg', frame)

            # SENDING AN EMAIL WITH THE INTRUDER'S IMAGE AS EMAIL ATTACHMENT
                # SET EMAIL LOGIN REQUIREMENTS
                # =============================================================================
                # =============================================================================
            gmail_user = 'christabellacosta01@gmail.com'
            gmail_app_password = 'ahbklodkddiwedhi'

            # SET THE INFO ABOUT THE SAID EMAIL

            sent_from = gmail_user
            sent_to = ['christabellacosta01@gmail.com', 'masterhomeowner@gmail.com']
            sent_subject = "INTRUDER ALERT!!!!"
            sent_body = ("an intruder was detected!!!\n\n"
                             "below is a captured image of the intruder\n\n"

                             " notification by your trusted home survaillence camera\n")

            # CREATE MESSAGE INSTANCE  AND ADD CONTENTS
            msg = MIMEMultipart()
            msg['From'] = sent_from
            msg['To'] = ",".join(sent_to)
            msg['Subject'] = sent_subject
            msg.attach(MIMEText(sent_body))

            with open(f'unknown{current_time}.jpg', 'rb') as f:
                # attach the image to the email
                img_data = MIMEImage(f.read())
                img_data.add_header('Content-Disposition', 'attachment', filename=f'unknown{current_time}.jpg')
                msg.attach(img_data)
                # =============================================================================

                try:
                    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                    server.login(gmail_user, gmail_app_password)
                    server.sendmail(sent_from, sent_to, msg.as_string())
                    server.close()

                    print('Email sent!')
                except Exception as exception:
                    print("Error: %s!\n\n" % exception)

            alert_sound()
    # ==================================================================================

    cv2.imshow('OUTPUT FRAME', frame)
    out.write(frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


