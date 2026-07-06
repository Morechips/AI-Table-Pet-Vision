import cv2
import mediapipe as mp
import csv

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# 打开存放数据的文件
csv_file = open('gesture_dataset.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)

# 特征归一化提取函数
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
    def normalize(n):
        return n / max_value if max_value != 0 else 0
        
    landmark_list = list(map(normalize, landmark_list))
    return landmark_list

# ⚠️ 注意：如果你有多个摄像头，记得修改这里的数字 (0 或 1)
# 💡 去掉了 cv2.CAP_DSHOW，防止部分 USB 摄像头驱动不兼容导致闪退
cap = cv2.VideoCapture(0) 

print("="*40)
print("📸 数据采集 2.0 启动！请按住对应数字键录制数据：")
print("[0] - 负样本 (乱动、自然下垂)")
print("[1] - 张开手 (挥手预备)")
print("[2] - 握紧拳头 (挥拳预备)")
print("[3] - 比心 (害羞)")
print("[4] - 伸出食指 (好奇/警觉)")
print("[q] - 退出采集")
print("="*40)

counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        print("⚠️ 无法读取画面，请检查摄像头连接！")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # 💡 【关键修复】把键盘监听移到外面！无论画面里有没有手，都能正常检测按键
    key = cv2.waitKey(1) & 0xFF
    label = -1
    
    if key == ord('0'): label = 0
    elif key == ord('1'): label = 1
    elif key == ord('2'): label = 2
    elif key == ord('3'): label = 3
    elif key == ord('4'): label = 4
    elif key == ord('q'): break  # 按下 q 键直接退出

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 如果按下了有效数字键，就把特征和标签写进表格
            if label != -1:
                features = process_landmarks(hand_landmarks)
                features.append(label)
                csv_writer.writerow(features)
                counts[label] += 1
                print(f"✅ 已采集 -> 标签 [{label}] | 当前总数: {counts[label]}")

    # 屏幕上显示录制进度
    y_pos = 30
    for k, v in counts.items():
        cv2.putText(frame, f"Label {k}: {v}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30

    cv2.imshow('Data Collection 2.0', frame)

# 释放资源
cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"🎉 采集结束！共收集数据：{sum(counts.values())} 条。")