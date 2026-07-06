import cv2
import mediapipe as mp
import pickle
import time
import serial
import math
import pandas as pd  
import collections 

# ======= 引入中文字体支持库 =======
try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ================= 配置区 =================
MODEL_PATH = 'gesture_model.pkl' 

COM_PORT = 'COM3'       # ⚠️ 请改成你的实际端口号
BAUD_RATE = 115200      

COOLDOWN_TIME = 2.0              
CONFIDENCE_THRESHOLD = 0.80      
WAVE_THRESHOLD = 0.15            
PUNCH_THRESHOLD = 0.08           
STATIC_HOLD_FRAMES = 10          

CMD_NONE  = 0x00
CMD_WAVE  = 0x01
CMD_PUNCH = 0x04
CMD_POINT = 0x05
CMD_HEART = 0x06

STATUS_NO_HAND = 101    
# =========================================

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"✅ 成功连接到串口 {COM_PORT}")
except Exception as e:
    print(f"❌ 串口连接失败: {e}")
    ser = None

def send_to_mcu(cmd_hex, track_x):
    """发送 4 字节协议包"""
    if ser:
        track_x_byte = track_x & 0xFF 
        packet = bytearray([0xAA, cmd_hex, track_x_byte, 0xFF])
        try:
            ser.write(packet)
        except Exception as e:
            pass

def put_chinese_text(img, text, position, text_color=(0, 255, 0), font_size=30):
    if not HAS_PIL:
        cv2.putText(img, "Detected", position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        return img
    try:
        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im)
        draw = ImageDraw.Draw(pil_im)
        try:
            font = ImageFont.truetype("msyh.ttc", font_size, encoding="utf-8")
        except:
            font = ImageFont.truetype("simhei.ttf", font_size, encoding="utf-8")
        draw.text(position, text, text_color, font=font)
        return cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    except:
        return img

with open(MODEL_PATH, 'rb') as f:
    clf = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def process_landmarks(hand_landmarks):
    landmark_list = []
    base_x = hand_landmarks.landmark[0].x
    base_y = hand_landmarks.landmark[0].y
    base_z = hand_landmarks.landmark[0].z
    for landmark in hand_landmarks.landmark:
        landmark_list.append(landmark.x - base_x)
        landmark_list.append(landmark.y - base_y)
        landmark_list.append(landmark.z - base_z)
    max_value = max(list(map(abs, landmark_list)))
    return [n / max_value if max_value != 0 else 0 for n in landmark_list]

cap = cv2.VideoCapture(0) 

print("="*40)
print("🤖 桌宠 [空闲静默 + 动作追踪] 已启动！")
print("="*40)

last_trigger_time = 0
last_serial_time = 0

wrist_x_history = collections.deque(maxlen=15) 
wrist_z_history = collections.deque(maxlen=15) 
heart_hold_frames = 0                              
pointing_hold_frames = 0                       

has_hand_last_frame = False 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()
    in_cooldown = (current_time - last_trigger_time) < COOLDOWN_TIME
    
    current_gesture = CMD_NONE
    track_x_val = STATUS_NO_HAND
    text_y_pos = 90 

    if results.multi_hand_landmarks:
        # ==========================================
        # 🟢 状态：检测到手
        # ==========================================
        has_hand_last_frame = True 
        
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # --- 空间追踪 ---
        palm_x = hand_landmarks.landmark[9].x
        track_x_val = int((0.5 - palm_x) * 200)
        track_x_val = max(-100, min(100, track_x_val)) 
        
        cv2.putText(frame, f"Track X (L/R): {track_x_val}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.line(frame, (int(w/2), 45), (int(w/2 - track_x_val*3), 45), (0, 255, 0), 6)
        
        # --- 动作识别 ---
        current_x = hand_landmarks.landmark[0].x
        current_depth = math.hypot(hand_landmarks.landmark[0].x - hand_landmarks.landmark[9].x, 
                                   hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y)

        features = process_landmarks(hand_landmarks)
        feature_names = [f'v{i}' for i in range(1, 64)]
        features_df = pd.DataFrame([features], columns=feature_names)

        probabilities = clf.predict_proba(features_df)[0]
        prediction = clf.predict(features_df)[0]
        
        if max(probabilities) >= CONFIDENCE_THRESHOLD and not in_cooldown:
            if prediction == 1:
                wrist_x_history.append(current_x)
                if len(wrist_x_history) == 15:
                    if (max(wrist_x_history) - min(wrist_x_history)) > WAVE_THRESHOLD:
                        current_gesture = CMD_WAVE
                        print(f"👋 发送指令: 0x01 (挥手)")
                        frame = put_chinese_text(frame, "✅ 识别：挥手", (10, text_y_pos), (0, 255, 0), 40)
                        last_trigger_time = current_time
                        wrist_x_history.clear()
                    else:
                        frame = put_chinese_text(frame, "张开手 (准备挥手)", (10, text_y_pos), (0, 255, 255), 30)
            
            elif prediction == 2:
                wrist_z_history.append(current_depth)
                if len(wrist_z_history) == 15:
                    if (max(wrist_z_history) - min(wrist_z_history)) > PUNCH_THRESHOLD:
                        current_gesture = CMD_PUNCH
                        print(f"✊ 发送指令: 0x04 (出拳)")
                        frame = put_chinese_text(frame, "✅ 识别：出拳", (10, text_y_pos), (0, 0, 255), 40)
                        last_trigger_time = current_time
                        wrist_z_history.clear()
                    else:
                        frame = put_chinese_text(frame, "握紧拳 (准备出拳)", (10, text_y_pos), (0, 165, 255), 30)
            
            elif prediction == 3:
                heart_hold_frames += 1
                if heart_hold_frames > STATIC_HOLD_FRAMES:
                    current_gesture = CMD_HEART
                    print("❤️ 发送指令: 0x06 (比心)")
                    frame = put_chinese_text(frame, "✅ 识别：比心", (10, text_y_pos), (255, 0, 255), 40)
                    last_trigger_time = current_time
                    heart_hold_frames = 0
                else:
                    frame = put_chinese_text(frame, f"比心识别中... {heart_hold_frames}/{STATIC_HOLD_FRAMES}", (10, text_y_pos), (255, 105, 180), 30)
            
            elif prediction == 4:
                pointing_hold_frames += 1
                if pointing_hold_frames > STATIC_HOLD_FRAMES:
                    current_gesture = CMD_POINT
                    print("☝️ 发送指令: 0x05 (指着)")
                    frame = put_chinese_text(frame, "✅ 识别：指着", (10, text_y_pos), (255, 255, 0), 40)
                    last_trigger_time = current_time
                    pointing_hold_frames = 0
                else:
                    frame = put_chinese_text(frame, f"指着屏幕中... {pointing_hold_frames}/{STATIC_HOLD_FRAMES}", (10, text_y_pos), (255, 255, 0), 30)
            else:
                wrist_x_history.clear(); wrist_z_history.clear()
                heart_hold_frames = 0; pointing_hold_frames = 0
                
        elif in_cooldown:
            frame = put_chinese_text(frame, f"动作冷却中... {COOLDOWN_TIME - (current_time - last_trigger_time):.1f}秒", (10, text_y_pos), (200, 200, 200), 30)
        else:
            frame = put_chinese_text(frame, "正常跟随中 (未触发特定动作)", (10, text_y_pos), (200, 200, 200), 30)

        # 🚀 【修改点】：只有在“有手”的时候，才以 50ms 频率发送串口！
        if current_time - last_serial_time > 0.05:
            send_to_mcu(current_gesture, track_x_val)
            last_serial_time = current_time

    else:
        # ==========================================
        # 🔴 状态：目标丢失 / 视野内无手
        # ==========================================
        wrist_x_history.clear(); wrist_z_history.clear()
        heart_hold_frames = 0; pointing_hold_frames = 0
        
        if not in_cooldown:
            frame = put_chinese_text(frame, "视野内无目标 (串口静默中)", (10, 90), (100, 100, 100), 30)
        
        # 🎯 边缘检测核心：只在手“离开的瞬间”发送一次断开指令，然后闭嘴
        if has_hand_last_frame:
            print(f"⚠️ 目标瞬间丢失！立刻发送断开指令 [AA 00 65 FF]")
            send_to_mcu(CMD_NONE, STATUS_NO_HAND) 
            has_hand_last_frame = False 
            
        # 注意：这里没有任何定时发送的代码！彻底释放 STM32 的接收中断！

    cv2.imshow('Table Pet UART Brain', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()