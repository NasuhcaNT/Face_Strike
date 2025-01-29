import cv2
import mediapipe as mp
import math
import random

# ---------------------------------------------------------
# 1) Parametreler
# ---------------------------------------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

ball_x = FRAME_WIDTH // 2
ball_y = FRAME_HEIGHT // 2
ball_radius = 20
ball_vx = 5
ball_vy = 5

hair_eyebrow_x = 0
hair_eyebrow_y = 0
hair_eyebrow_radius = 30
region_found = False

score = 0

# Kale
goal_width = 80
goal_height = 80
goal_x = FRAME_WIDTH - goal_width
goal_y = 0

# Kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("Kamera açılamadı!")
    exit()

# -- BURADA: Tam ekran pencereyi oluşturuyoruz --
window_name = "Face Ball"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        region_found = False
        biggest_area = 0

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x_min = int(bbox.xmin * FRAME_WIDTH)
                y_min = int(bbox.ymin * FRAME_HEIGHT)
                w_box = int(bbox.width * FRAME_WIDTH)
                h_box = int(bbox.height * FRAME_HEIGHT)
                area = w_box * h_box
                if area <= 0:
                    continue

                keypoints = detection.location_data.relative_keypoints
                if len(keypoints) < 2:
                    continue

                right_eye = keypoints[0]
                left_eye = keypoints[1]

                right_eye_x = int(right_eye.x * FRAME_WIDTH)
                right_eye_y = int(right_eye.y * FRAME_HEIGHT)
                left_eye_x = int(left_eye.x * FRAME_WIDTH)
                left_eye_y = int(left_eye.y * FRAME_HEIGHT)

                eyes_center_y = (right_eye_y + left_eye_y) // 2
                EYEBROW_OFFSET = 15
                eyebrow_line_y = eyes_center_y - EYEBROW_OFFSET

                hair_line_y = y_min
                region_height = eyebrow_line_y - hair_line_y
                if region_height <= 0:
                    continue

                region_center_x = x_min + w_box // 2
                region_center_y = hair_line_y + region_height // 2
                tmp_radius = max(w_box, region_height) // 2

                if area > biggest_area:
                    biggest_area = area
                    region_found = True
                    hair_eyebrow_x = region_center_x
                    hair_eyebrow_y = region_center_y
                    hair_eyebrow_radius = tmp_radius

        # Top hareket
        ball_x += ball_vx
        ball_y += ball_vy

        collided_hair_eyebrow = False
        collided_edge_x = False
        collided_edge_y = False

        # Kenarlar
        if (ball_x - ball_radius) < 0:
            overlap = ball_radius - ball_x
            ball_x += overlap
            if not collided_edge_x:
                ball_vx = -ball_vx
                collided_edge_x = True
        
        if (ball_x + ball_radius) > FRAME_WIDTH:
            overlap = (ball_x + ball_radius) - FRAME_WIDTH
            ball_x -= overlap
            if not collided_edge_x:
                ball_vx = -ball_vx
                collided_edge_x = True

        if (ball_y - ball_radius) < 0:
            overlap = ball_radius - ball_y
            ball_y += overlap
            if not collided_edge_y:
                ball_vy = -ball_vy
                collided_edge_y = True

        if (ball_y + ball_radius) > FRAME_HEIGHT:
            overlap = (ball_y + ball_radius) - FRAME_HEIGHT
            ball_y -= overlap
            if not collided_edge_y:
                ball_vy = -ball_vy
                collided_edge_y = True

        # Kale kontrol
        if (goal_x <= ball_x <= goal_x + goal_width) and \
           (goal_y <= ball_y <= goal_y + goal_height):
            score += 1
            ball_x = random.randint(ball_radius, FRAME_WIDTH - ball_radius)
            ball_y = random.randint(ball_radius, FRAME_HEIGHT - ball_radius)

        # Saç-Kaş çarpışma
        if region_found:
            dx = ball_x - hair_eyebrow_x
            dy = ball_y - hair_eyebrow_y
            distance = math.sqrt(dx**2 + dy**2)
            if distance < (ball_radius + hair_eyebrow_radius):
                overlap = (ball_radius + hair_eyebrow_radius) - distance
                if distance != 0:
                    nx = dx / distance
                    ny = dy / distance
                    ball_x += nx * overlap
                    ball_y += ny * overlap
                
                if not collided_hair_eyebrow:
                    ball_vx = -ball_vx
                    ball_vy = -ball_vy
                    collided_hair_eyebrow = True

        # Çizimler
        if region_found:
            cv2.circle(frame, (hair_eyebrow_x, hair_eyebrow_y), hair_eyebrow_radius, (0, 255, 0), 2)
            cv2.putText(frame, "Face Found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(frame, "Face Not Found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Top
        cv2.circle(frame, (int(ball_x), int(ball_y)), ball_radius, (0, 0, 255), -1)

        # Kale
        cv2.rectangle(frame, (goal_x, goal_y), (goal_x+goal_width, goal_y+goal_height),
                      (255, 255, 0), 2)
        cv2.putText(frame, "Goal", (goal_x+5, goal_y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Skor
        cv2.putText(frame, f"Skor: {score}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # -- BURADA: Göstermek istediğimiz pencere ismi "window_name" --
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
