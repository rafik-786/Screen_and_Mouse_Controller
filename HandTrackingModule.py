# Dependencies
import cv2
import mediapipe as mp
import time
import math
import HardCodeModule as hc


class handDetector:

    def __init__(self, mode=False, maxHands=1, detectComplex=1, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectComplex = detectComplex
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.handProcessed = None

    def findHand(self, img, draw=True):

        # Mediapipe works on RGB images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mediapipe is detecting the hand
        self.handProcessed = self.hands.process(imgRGB)

        if self.handProcessed.multi_hand_landmarks:

            # handLMS contains [X-axis,Y-axis]

            for handLMS in self.handProcessed.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHands.HAND_CONNECTIONS)

        return img

    def findCoordinates(self, img, handNo=0, draw=False):

        # lamMarkList contains [LandMark Id,ID-x-axis,ID-y-axis]
        landMarkList = []
        landmarkFound = False

        if self.handProcessed.multi_hand_landmarks:
            theHand = self.handProcessed.multi_hand_landmarks[handNo]

            for ID, lm in enumerate(theHand.landmark):
                height, width = img.shape[0], img.shape[1]

                # converting to image width and height
                cx, cy = int(lm.x * width), int(lm.y * height)

                landMarkList.append([ID, cx, cy])
                radiusLandMark = 2
                thicknessLandMark = 3
                if draw:
                    cv2.circle(img, (cx, cy), radiusLandMark, hc.COLOR_BLUE, thicknessLandMark)

            if len(landMarkList) != 0:
                landmarkFound = True

        return landMarkList, landmarkFound

    def isHandStraight(self, img, draw=False):
        landmarkList, landmarkFound = handDetector.findCoordinates(self, img)
        AXIS = (25, 50)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        fontSCALE = 1
        THICKNESS = 1
        # if draw:
        #     cv2.putText(img, "Not Straight", AXIS, FONT, fontSCALE, hc.COLOR_GREEN, THICKNESS)
        # return False

        if landmarkFound:
            if landmarkList[hc.MIDDLE_PIP][hc.y_axis] < landmarkList[hc.THUMB_MCP][hc.y_axis]:
                if landmarkList[hc.MIDDLE_PIP][hc.y_axis] < landmarkList[hc.PINKY_MCP][hc.y_axis]:
                    if draw:
                        cv2.putText(img, "Straight", AXIS, FONT, fontSCALE, hc.COLOR_GREEN, THICKNESS)
                    return True
                else:
                    if draw:
                        cv2.putText(img, "Not Straight", AXIS, FONT, fontSCALE, hc.COLOR_GREEN, THICKNESS)
                    return False
            else:
                if draw:
                    cv2.putText(img, "Not Straight", AXIS, FONT, fontSCALE, hc.COLOR_GREEN, THICKNESS)
                return False


    def checkHand(self, img, draw=False):

        landMarkList, landmarkFound = handDetector.findCoordinates(self, img)

        leftHand = False
        rightHand = False
        handIsStraight = handDetector.isHandStraight(self, img)
        if landmarkFound:
            if handIsStraight:
                # do something
                thumb = landMarkList[hc.THUMB_TIP][hc.x_axis]
                pinky = landMarkList[hc.PINKY_TIP][hc.x_axis]
                leftHand = thumb < pinky


            else:
                wrist = landMarkList[hc.WRIST][hc.x_axis]
                thumb = landMarkList[hc.THUMB_TIP][hc.x_axis]
                leftHand = thumb < wrist

            rightHand = not leftHand

            if draw:
                AXIS = (5, 50)
                FONT = cv2.FONT_HERSHEY_SIMPLEX
                fontSCALE = 1
                THICKNESS = 1
                if leftHand:
                    cv2.putText(img, "Left Hand", AXIS, FONT, fontSCALE, hc.COLOR_GREEN, THICKNESS)
                else:
                    cv2.putText(img, "Right Hand", AXIS, FONT, fontSCALE, hc.COLOR_GREEN, THICKNESS)
            return [leftHand, rightHand]

    def fingersUP(self, img):

        landMarkList, landmarkFound = handDetector.findCoordinates(self, img)

        fingersuplist = []
        tipsList = [hc.INDEX_TIP, hc.MIDDLE_TIP, hc.RING_TIP, hc.PINKY_TIP]
        if landmarkFound:
            leftHand, rightHand = handDetector.checkHand(self, img)

            if handDetector.isHandStraight(self, img):
                # fingers contains which fingers are up

                if rightHand:
                    fingersuplist.append(landMarkList[hc.THUMB_TIP][hc.x_axis] > landMarkList[hc.THUMB_IP][hc.x_axis])
                else:
                    fingersuplist.append(landMarkList[hc.THUMB_TIP][hc.x_axis] < landMarkList[hc.THUMB_IP][hc.x_axis])

                for tip in tipsList:
                    fingersuplist.append(landMarkList[tip][hc.y_axis] < landMarkList[tip - 2][hc.y_axis])
            else:
                # thumb
                fingersuplist.append(landMarkList[hc.INDEX_TIP][hc.y_axis] - landMarkList[hc.THUMB_IP][hc.y_axis] > 45)

                if rightHand:

                    for tip in tipsList:
                        fingersuplist.append(landMarkList[tip][hc.x_axis] > landMarkList[tip - 2][hc.x_axis])
                else:

                    for tip in tipsList:
                        fingersuplist.append(landMarkList[tip][hc.x_axis] < landMarkList[tip - 2][hc.x_axis])

        return fingersuplist

    def landmarkCoordinates(self, img, landmark):
        landmarkList, landmarkFound = handDetector.findCoordinates(self, img)
        if landmarkFound:
            return landmarkList[landmark][hc.x_axis], landmarkList[landmark][hc.y_axis]
        else:
            return 0, 0

    def landmarkDistance(self, img, landmark1, landmark2):

        x1, y1 = handDetector.landmarkCoordinates(self, img, landmark1)
        x2, y2 = handDetector.landmarkCoordinates(self, img, landmark2)

        # calculate distance

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # draw line
        point1 = (x1, y1)
        point2 = (x2, y2)
        thickness = 2
        cv2.line(img, point1, point2, hc.COLOR_RED, thickness)

        # draw circle
        center = (cx, cy)
        radius1 = 6
        radius2 = 4
        cv2.circle(img, center, radius1, hc.COLOR_BLACK, cv2.FILLED)
        cv2.circle(img, center, radius2, hc.COLOR_BROWN, cv2.FILLED)

        return distance, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)

    detector = handDetector()
    while True:
        success, img = cap.read()

        img = detector.findHand(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # lmlist = detector.findCoordinates(img)
        # checkHand = detector.checkHand(img, draw=True)
        fingers = detector.fingersUP(img)
        # print(fingers)
        # print(detector.landmarkCoordinates(img, 8))
        # detector.landmarkDistance(img,hc.INDEX_TIP,hc.MIDDLE_TIP)
        # print(fingers)
        # print(checkHand)
        # print(detector.isHandStraight(img, draw=True))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow("Vide", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
