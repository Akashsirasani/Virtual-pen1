# Virtual-pen1
import cv2
import numpy as np
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, maxHands=1, detectionCon=0.7, trackCon=0.6):
        self.maxHands = maxHands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=self.maxHands,
            model_complexity=1,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0):
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            h, w, _ = img.shape
            for i, lm in enumerate(hand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((i, cx, cy))
        return lmList

    def fingersUp(self, lmList):
        fingers = []
        if not lmList:
            return [0] * 5
        fingers.append(1 if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1] else 0)
        for id in range(1, 5):
            fingers.append(1 if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2] else 0)
        return fingers

# ======================== Virtual Pen App ========================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

canvas = np.zeros((720, 1280, 3), np.uint8)
detector = HandDetector()

colors = [(255, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 128, 255), (255, 255, 0), (255, 128, 0)]
drawColor = colors[0]
eraserColor = (0, 0, 0)

brushThickness = 7
eraserThickness = 60

xp, yp = 0, 0
isEraser = False
smoothFactor = 0.5  # Increased smoothing to reduce motion jitter
drawingAllowed = False

colorBarHeight = 100  # Prevent drawing in the color selection area
motionThreshold = 50  # Increased threshold to slow down the motion

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Draw color selection bar
    for i, color in enumerate(colors):
        cv2.rectangle(img, (i * 100 + 10, 10), (i * 100 + 100, 100), color, cv2.FILLED)
    cv2.rectangle(img, (len(colors) * 100 + 10, 10), (len(colors) * 100 + 100, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "Eraser", (len(colors) * 100 + 20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if lmList:
        fingers = detector.fingersUp(lmList)
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # Clear canvas with all fingers up
        if fingers == [1, 1, 1, 1, 1]:
            canvas = np.zeros((720, 1280, 3), np.uint8)
            cv2.putText(img, "Canvas Cleared", (500, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)

        # Selection Mode (Two fingers up)
        elif fingers[1] and fingers[2]:
            xp, yp = 0, 0
            drawingAllowed = False
            if y1 < colorBarHeight:
                for i in range(len(colors)):
                    if i * 100 + 10 < x1 < i * 100 + 100:
                        drawColor = colors[i]
                        isEraser = False
                if len(colors) * 100 + 10 < x1 < len(colors) * 100 + 100:
                    isEraser = True
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), (0, 0, 0), cv2.FILLED)

        # Drawing Mode (Index finger up)
        elif fingers == [0, 1, 0, 0, 0]:
            if y1 > colorBarHeight:
                if not drawingAllowed:
                    xp, yp = x1, y1
                    drawingAllowed = True

                smoothX = int((1 - smoothFactor) * xp + smoothFactor * x1)
                smoothY = int((1 - smoothFactor) * yp + smoothFactor * y1)

                dist = math.hypot(smoothX - xp, smoothY - yp)  # Calculate the distance moved
                if dist < motionThreshold:  # Only draw if the movement is small enough (slower motion)
                    if isEraser:
                        cv2.line(canvas, (xp, yp), (smoothX, smoothY), eraserColor, eraserThickness, lineType=cv2.LINE_AA)
                    else:
                        cv2.line(canvas, (xp, yp), (smoothX, smoothY), drawColor, brushThickness, lineType=cv2.LINE_AA)

                xp, yp = smoothX, smoothY
            else:
                drawingAllowed = False
                xp, yp = 0, 0
        else:
            drawingAllowed = False
            xp, yp = 0, 0

    # Merge canvas with webcam image
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY)
    maskInv = cv2.bitwise_not(mask)
    maskInv = cv2.cvtColor(maskInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, maskInv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Virtual Pen", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
