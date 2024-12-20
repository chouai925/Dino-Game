import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import pyautogui
import time
import wc_gestureCNN_torch as myNN
from PIL import Image
import json
import threading
import os
import webbrowser

# 初始化參數
minValue = 70
x0 = 400
y0 = 200
height = 200
width = 200

saveImg = False
guessGesture = True
visualize = False
quietMode = False
takebkgrndSubMask = False
gestname = ''
path = './'

# 初始化內核
kernel = np.ones((15,15), np.uint8)
kernel2 = np.ones((1,1), np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# 遮罩模式
binaryMode = True
bkgrndSubMode = False
mask = 0
bkgrnd = 0
isBinary = True
prev_gray = None

# 手勢到動作的映射
gesture_actions = {
    0: ("OK", None),        # OK - 無動作
    1: ("NOTHING", None),   # NOTHING - 無動作
    2: ("PEACE", None),     # PEACE - 無動作
    3: ("PUNCH", "space"),  # PUNCH - 跳躍
    4: ("STOP", "down")     # STOP - 蹲下
}

def init_dino_game():
    try:
        webbrowser.open('https://chromedino.com/')
        print("Opening Chrome Dino game...")
        time.sleep(2)  # 等待遊戲加載
        pyautogui.press('space')  # 開始遊戲
        time.sleep(1)  # 等待遊戲開始
    except Exception as e:
        print(f"Error initializing dino game: {e}")

def skinMask(frame, x0, y0, width, height, framecount, plot):
    """皮膚檢測遮罩"""
    global guessGesture, mod, visualize
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    # blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    
    # bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    if guessGesture and (framecount % 5) == 4:
        t = threading.Thread(target=myNN.guess_gesture, args=[mod, res])
        t.start()
    elif visualize:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    return res

def binaryMask(frame, x0, y0, width, height, framecount, plot):
    """二值化遮罩"""
    global guessGesture, mod, visualize, isBinary, prev_gray
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 2)
    
    if isBinary:
        # 二值化模式
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,2)
        ret, res = cv2.threshold(th3, 30, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        
        # 形態學操作
        kernel = np.ones((3,3),np.uint8)
        res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=1)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel, iterations=1)
    else:
        # 非二值化模式 - 使用運動檢測
        if prev_gray is None:
            prev_gray = blur
            return blur
        
        # 計算當前幀與前一幀的差異
        frame_diff = cv2.absdiff(blur, prev_gray)
        
        # 應用閾值處理
        _, motion_mask = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)
        
        # 應用形態學操作來改善遮罩
        kernel = np.ones((5,5), np.uint8)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        motion_mask = cv2.erode(motion_mask, kernel, iterations=1)
        
        # 使用運動遮罩來提取手部區域
        res = cv2.bitwise_and(blur, blur, mask=motion_mask)
        
        # 增強對比度
        res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
        
        # 更新前一幀
        prev_gray = blur.copy()
        
        # 如果檢測到的運動區域太小，使用原始模糊圖像
        if cv2.countNonZero(motion_mask) < 100:
            res = blur
    
    if guessGesture and (framecount % 5) == 4:
        t = threading.Thread(target=myNN.guess_gesture, args=[mod, res])
        t.start()
    elif visualize:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(1)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    return res

def bkgrndSubMask(frame, x0, y0, width, height, framecount, plot):
    """背景減除遮罩"""
    global guessGesture, takebkgrndSubMask, visualize, mod, bkgrnd
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Take background image
    if takebkgrndSubMask:
        bkgrnd = roi
        takebkgrndSubMask = False
        print("Refreshing background image for mask...")
    
    if bkgrnd is None or isinstance(bkgrnd, int):
        bkgrnd = roi
        return roi
    
    # Take a diff between roi & bkgrnd image contents
    diff = cv2.absdiff(roi, bkgrnd)
    
    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    mask = cv2.GaussianBlur(diff, (3,3), 5)
    mask = cv2.erode(diff, skinkernel, iterations = 1)
    mask = cv2.dilate(diff, skinkernel, iterations = 1)
    res = cv2.bitwise_and(roi, roi, mask = mask)
    
    if guessGesture and (framecount % 5) == 4:
        t = threading.Thread(target=myNN.guess_gesture, args=[mod, res])
        t.start()
    elif visualize:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    return res

def saveROIImg(img):
    """保存 ROI 圖像用於訓練"""
    global path, gestname, saveImg
    if not os.path.exists(gestname):
        os.makedirs(gestname)
    
    full_path = os.path.join(path, gestname)
    img_count = len(os.listdir(full_path))
    
    save_path = os.path.join(full_path, f"{gestname}_{img_count}.png")
    cv2.imwrite(save_path, img)
    print(f"Saved image: {save_path}")

def Main():
    """主程序"""
    global guessGesture, mod, binaryMode, bkgrndSubMode, mask, bkgrnd, x0, y0, gestname, saveImg, path, quietMode, takebkgrndSubMask

    # 初始化恐龍遊戲
    init_dino_game()

    # 設置 pyautogui 的安全設置
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1  # 設置操作間隔時間

    # 字體設置
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    fx = 10
    fy = 350
    fh = 18
    
    # 初始化攝像頭
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # 加載預訓練模型
    print("Loading gesture recognition model...")
    mod = myNN.load_model()
    if mod is None:
        print("Error loading model")
        return
    
    mod = mod.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    mod.eval()

    # 設置攝像頭分辨率
    ret = cap.set(3, 640)
    ret = cap.set(4, 480)

    framecount = 0
    fps = ""
    start = time.time()

    plot = np.zeros((512,512,3), np.uint8)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 3)
            
            if framecount == 0:
                print("Starting predictions...")
            
            # 計算 FPS
            if framecount % 20 == 0:
                end = time.time()
                fps = "FPS: {:.1f}".format(20/(end-start))
                start = time.time()
            
            if bkgrndSubMode:
                if takebkgrndSubMask:
                    bkgrnd = frame
                    takebkgrndSubMask = False
                mask = bkgrndSubMask(frame, x0, y0, width, height, framecount, plot)
            elif binaryMode:
                mask = binaryMask(frame, x0, y0, width, height, framecount, plot)
            else:
                mask = skinMask(frame, x0, y0, width, height, framecount, plot)
            
            # 保存訓練圖像
            if saveImg:
                saveROIImg(mask)
                
            # 顯示 FPS 和預測結果
            cv2.putText(frame, fps, (10,20), font, 0.7, (0,255,0), 2, 1)
            cv2.putText(frame, f"Prediction: {myNN.current_gesture}", (10,80),
                       font, 0.7, (0,255,0), 2, 1)
            
            # 顯示選項菜單
            if not quietMode:
                cv2.putText(frame, 'Options:', (fx,fy), font, 0.7, (0,255,0), 2, 1)
                cv2.putText(frame, 'b - Toggle Binary/SkinMask', (fx,fy + fh), font, size, (0,255,0), 1, 1)
                cv2.putText(frame, 'x - Toggle Background Sub Mask', (fx,fy + 2*fh), font, size, (0,255,0), 1, 1)
                cv2.putText(frame, 'g - Toggle Prediction Mode', (fx,fy + 3*fh), font, size, (0,255,0), 1, 1)
                cv2.putText(frame, 'q - Toggle Quiet Mode', (fx,fy + 4*fh), font, size, (0,255,0), 1, 1)
                cv2.putText(frame, 'n - Enter name for new gesture', (fx,fy + 5*fh), font, size, (0,255,0), 1, 1)
                cv2.putText(frame, 's - Start/Stop saving images', (fx,fy + 6*fh), font, size, (0,255,0), 1, 1)
                cv2.putText(frame, 'ESC - Exit', (fx,fy + 7*fh), font, size, (0,255,0), 1, 1)
            
            # 執行預測的手勢動作
            gesture_index = myNN.output.index(myNN.current_gesture) if myNN.current_gesture in myNN.output else 1
            if gesture_index in gesture_actions and not saveImg:  # 在訓練模式下不執行動作
                gesture_name, action_key = gesture_actions[gesture_index]
                if action_key:
                    pyautogui.press(action_key)
            
            # 顯示窗口
            if not quietMode:
                cv2.imshow('Original', frame)
                cv2.imshow('ROI', mask)
            
            # 鍵盤控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('b'):
                binaryMode = not binaryMode
                bkgrndSubMode = False
                print("Binary Threshold filter - {}".format(binaryMode))
            elif key == ord('x'):
                takebkgrndSubMask = True
                bkgrndSubMode = True
                print("Background Subtraction filter - {}".format(bkgrndSubMode))
            elif key == ord('g'):
                guessGesture = not guessGesture
                print("Prediction Mode - {}".format(guessGesture))
            elif key == ord('q'):
                quietMode = not quietMode
                print("Quiet Mode - {}".format(quietMode))
            elif key == ord('n'):
                gestname = input("Enter the gesture folder name: ")
                try:
                    os.makedirs(gestname)
                except OSError as e:
                    if e.errno != 17:  # 17 是"文件已存在"的錯誤碼
                        print('Error creating directory: ' + gestname)
                        gestname = ''
                    else:
                        print('Directory already exists: ' + gestname)
                
                if gestname:
                    path = os.path.join('.', gestname)
            elif key == ord('s'):
                if not gestname:
                    print("Enter gesture folder name first (press 'n')")
                else:
                    saveImg = not saveImg
                    print("Save Images - {}".format(saveImg))
            elif key == ord('i'):
                y0 = max(0, y0 - 5)
            elif key == ord('k'):
                y0 = min(frame.shape[0] - height, y0 + 5)
            elif key == ord('j'):
                x0 = max(0, x0 - 5)
            elif key == ord('l'):
                x0 = min(frame.shape[1] - width, x0 + 5)
            
            framecount += 1
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        Main()
    except Exception as e:
        print(f"Error: {str(e)}")
