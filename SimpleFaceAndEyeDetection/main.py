import PySimpleGUI as sg
import cv2

# window
layout           = [[sg.Image(key = 'FRAME')]]
window           = sg.Window('OpenCV Face & Eye Detector', layout)

# variables
video            = cv2.VideoCapture(0)
face_classifier  = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
eye_classifier   = cv2.CascadeClassifier("data/haarcascade_eye.xml")
box_colour       = (0,255,0)
box_border       = 1
eye_box_adjuster = 12                         
    
# main loop   
while True:
    
    # check for events in the window
    event, values = window.read(timeout = 0)
    
    # when a windows close event has been detected exit main loop
    if event == sg.WIN_CLOSED:
        break
    
    # read data for colour and grayscale frames
    _, colour_frame  = video.read()
    grayscale_frame  = cv2.cvtColor(colour_frame, cv2.COLOR_BGR2GRAY)

    # identify faces
    faces = face_classifier.detectMultiScale(grayscale_frame,
                                             scaleFactor  = 1.1,
                                             minNeighbors = 4,
                                             minSize      = (20,20),
                                             flags        = cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces: 
        cv2.rectangle(colour_frame, (x , y), (x + w, y + h), box_colour, box_border) 

    # identify eyes
    eyes = eye_classifier.detectMultiScale(  grayscale_frame,
                                             scaleFactor  = 1.2,
                                             minNeighbors = 4,
                                             minSize      = (3, 3),
                                             flags        = cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in eyes:     
        cv2.rectangle(colour_frame,
                      (x + eye_box_adjuster,     y + eye_box_adjuster),     # top lefthand corner
                      ((x + w)-eye_box_adjuster, (y + h)-eye_box_adjuster), # bottom righthand corner
                      box_colour,                                   
                      box_border)  
     
    # update the frame image
    image_bytes = cv2.imencode('.png', colour_frame)[1].tobytes()
    window['FRAME'].update(data = image_bytes)

# stop    
window.close()