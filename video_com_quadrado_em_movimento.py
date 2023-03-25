
# leia com atenção!!! este código roda em sua máquina local.

import cv2
import numpy as np

# carrega o video
cap = cv2.VideoCapture('people-walking.mp4')

# Cria a subtração do fundo
#fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorKNN()

while (1):
    ret, frame = cap.read()

    if not ret:
        break
    # Aplica a mascara no frame recebido
    fgmask = fgbg.apply(frame)

    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    dilatation = cv2.dilate(opening, kernel, iterations=9)

    contours, hierarchy = cv2.findContours(
        dilatation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (16, 216, 224), 2)

    cv2.imshow('fgmask', frame)
    # cv2.imshow('frame',fgmask)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
