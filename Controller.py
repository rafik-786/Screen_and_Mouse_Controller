# Dependencies

import cv2
import time
import numpy as np
import pyautogui as cntrl

import HardCodeModule as hc
import HandTrackingModule as htm

# Capture Video
cap = cv2.VideoCapture(0)

# Set Camera Property
wCam = 640
hCam = 480

cap.set(hc.Camera_width, wCam)
cap.set(hc.Camera_Height, hCam)

# Get Screen Property
wScreen, hScreen = cntrl.size()

# Create HandDetector Object
detector = htm.handDetector()

# **********Variables**************

# Frame
pTime = 0

# Font
FONT1 = cv2.FONT_HERSHEY_SIMPLEX

# Track Area
x_start_Frame, y_start_Frame = 50, 10
x_end_Frame, y_end_Frame = 50, 150

# Screen Locations
cLocX, cLocY, pLocX, pLocY = 0, 0, 0, 0

# Cursor Movement Smooth value
smoothVal = 2.75

# window Property
maximize = False
minimize = False

# Desktop lock
lock_desktop = False
switch = False
switch_ready = False
# *********************************

# ***********Functions*************


# *********************************


while True:
    # Read Frame
    code, img = cap.read()

    # Detect Hand
    img = detector.findHand(img)

    # Draw Track Area
    trackSPoint = (x_start_Frame, y_start_Frame)
    trackEPoint = (wCam - x_end_Frame, hCam - y_end_Frame)
    trackThickness = 2
    cv2.rectangle(img, trackSPoint, trackEPoint, hc.COLOR_LAVENDER, trackThickness)

    handIsStraight = detector.isHandStraight(img)

    # find the coordinates of the all landmarks
    lmList, lmFound = detector.findCoordinates(img)

    if lmFound:
        # find all the fingers that are up
        fingers = detector.fingersUP(img)

        # ###################################Cursor movement using Index Tip############################################
        if not fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            x, y = detector.landmarkCoordinates(img, hc.INDEX_TIP)

            # draw circle
            radius1 = 10
            radius2 = 8
            cv2.circle(img, (x, y), radius1, hc.COLOR_BLACK, cv2.FILLED)
            cv2.circle(img, (x, y), radius2, hc.COLOR_LAVENDER, cv2.FILLED)

            # interpret Track axis coordinates to screen axis coordinates
            x_mouse = np.interp(x, (x_start_Frame, wCam - x_end_Frame), (0, wScreen))
            y_mouse = np.interp(y, (y_start_Frame, hCam - y_end_Frame), (0, hScreen))

            # smoothen cursor movement
            cLocX = pLocX + (x_mouse - pLocX) / smoothVal
            cLocY = pLocY + (y_mouse - pLocY) / smoothVal
            pLocX, pLocY = cLocX, cLocY

            # move cursor
            cntrl.moveTo(wScreen - x_mouse, y_mouse)  # Frame Drop

        # ########################################Cursor left click using Middle Tip####################################
        if not fingers[0] and fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            x1, y1 = detector.landmarkCoordinates(img, hc.MIDDLE_TIP)
            x2, y2 = detector.landmarkCoordinates(img, hc.INDEX_TIP)

            # draw circle for Index and Middle
            # index
            radius1 = 10
            radius2 = 8
            cv2.circle(img, (x1, y1), radius1, hc.COLOR_BLACK, cv2.FILLED)
            cv2.circle(img, (x1, y1), radius2, hc.COLOR_LAVENDER, cv2.FILLED)
            # middle
            radius1 = 10
            radius2 = 8
            cv2.circle(img, (x2, y2), radius1, hc.COLOR_BLACK, cv2.FILLED)
            cv2.circle(img, (x2, y2), radius2, hc.COLOR_LAVENDER, cv2.FILLED)

            # find distance b/w index and middle and get lineInfo
            distance, lineInfo = detector.landmarkDistance(img, hc.INDEX_TIP, hc.MIDDLE_TIP)

            if distance < 45:
                # draw circle if distance is less than 45
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 6, hc.COLOR_GREEN, cv2.FILLED)

                # Left mouse click
                cntrl.click()

        # ####################################Cursor right click with Pinky Tip and Index Tip###########################
        if not fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:
            x, y = detector.landmarkCoordinates(img, hc.PINKY_TIP)

            # draw circle
            radius1 = 10
            radius2 = 8
            cv2.circle(img, (x, y), radius1, hc.COLOR_BLACK, cv2.FILLED)
            cv2.circle(img, (x, y), radius2, hc.COLOR_LAVENDER, cv2.FILLED)

            # right mouse click
            cntrl.rightClick()

        # ********************************************Maximize and Minimize windows*************************************
        # #################################################Maximize with all tips#######################################
        if handIsStraight and not maximize and fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            maximize = True
            minimize = False

            # control to maximize windows
            print("Maximized")
            cntrl.hotkey('win', 'up')

        # #############################################Minimize with index and thumb####################################
        if handIsStraight and not minimize and fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not \
                fingers[4]:
            maximize = False
            minimize = True
            print("Minimized")
            # control to minimize windows
            cntrl.hotkey('win', 'down')

        # ############################################Scrolling up and down############################################
        if handIsStraight and not fingers[0] and not fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            pink_TIP_TO_MCP_distance = detector.landmarkDistance(img, hc.PINKY_TIP, hc.PINKY_MCP)[0]
            ring_TIP_TO_MCP_distance = detector.landmarkDistance(img, hc.RING_TIP, hc.RING_MCP)[0]
            middle_TIP_TO_MCP_distance = detector.landmarkDistance(img, hc.MIDDLE_TIP, hc.MIDDLE_MCP)[0]

            # print(middle_TIP_TO_MCP_distance, " ", ring_TIP_TO_MCP_distance, " ", pink_TIP_TO_MCP_distance)

            if pink_TIP_TO_MCP_distance >= 60 and ring_TIP_TO_MCP_distance >= 85 and middle_TIP_TO_MCP_distance >= 90:
                cntrl.scroll(65)

            else:
                cntrl.scroll(-65)

        ############################################ Show Desktop ####################################################
        if handIsStraight and not fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:

            pink_distance = int(detector.landmarkDistance(img, hc.PINKY_TIP, hc.RING_DIP)[0])
            ring_distance = int(detector.landmarkDistance(img, hc.RING_TIP, hc.MIDDLE_TIP)[0])
            middle_distance = int(detector.landmarkDistance(img, hc.MIDDLE_TIP, hc.INDEX_TIP)[0])

            if 35 > pink_distance >= 15 and 40 > ring_distance >= 15 and 35 > middle_distance >= 15 and not lock_desktop:
                cntrl.hotkey('win', 'd')
                lock_desktop = True

            if 79 > pink_distance >= 36 and 78 > ring_distance >= 41 and 88 > middle_distance >= 36 and lock_desktop:
                lock_desktop = False

        ########################################### switch windows #####################################################

        if not handIsStraight and  fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not fingers[4] and not switch and not switch_ready:

            # activate for switching
            print("Switch Mode: Ready")
            switch = True

        elif not handIsStraight and not fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not \
                fingers[4] and switch and not switch_ready:

            # disabling for getting activated again
            switch = False
            print("Switching Mode Running")

            # activate for switching windows
            switch_ready = True
        elif not handIsStraight and fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and not \
                fingers[4] and not switch and switch_ready:

            # show available windows
            print("Switching..")
            cntrl.keyDown('alt')
            cntrl.press("tab")
        elif not handIsStraight and not fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not \
                fingers[4] and not switch and switch_ready:

            # select windows  and reset all
            print("Switching Paused")
            cntrl.press("enter")
            switch = False
            switch_ready = False
            cntrl.keyUp('alt')

            print("Switch Mode :Deactivated \n Reset Done")
            # ##############################################################################################################

    # show FPS
    cTime = time.time()
    FPS = 1 / (cTime - pTime)
    pTime = cTime
    pointFPS = (5, 20)
    cv2.putText(img, "Frame:" + str(int(FPS)), pointFPS, FONT1, 0.8, hc.COLOR_GREEN, 1)

    # Display Frame
    cv2.imshow("Application", img)

    # Close App
    if cv2.waitKey(1) == ord('q'):
        # Reset everything
        # window Property
        maximize = False
        minimize = False

        # Desktop lock
        lock_desktop = False
        switch = False
        switch_ready = False

        break
