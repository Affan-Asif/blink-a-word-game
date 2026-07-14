import cv2
import mediapipe as mp
from cvzone.FaceMeshModule import FaceMeshDetector
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = FaceMeshDetector(maxFaces=1)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
y_position = 360
sentence = "Welcome to the blink to select word game"
words = sentence.split()
blink_threshold = 1.5
blink_cooldown = 1.5
points_per_word = 10
bonus_points = 50
game_duration = 30
screen_width = 1280
word_gap = 40 

def get_eye_aspect_ratio(eye_points):
    top = eye_points[1]
    bottom = eye_points[5]
    left = eye_points[0]
    right = eye_points[3]
    vertical = ((top[0] - bottom[0])**2 + (top[1] - bottom[1])**2)**0.5
    horizontal = ((left[0] - right[0])**2 + (left[1] - right[1])**2)**0.5
    return horizontal / vertical if vertical != 0 else 0

def reset_game():
    global word_positions, selected_words, points, last_blink_time, start_time
    selected_words = []
    points = 0
    last_blink_time = 0
    start_time = time.time()
    word_positions = []
    current_x = screen_width
    for word in words:
        word_width = cv2.getTextSize(word, font, font_scale, 2)[0][0]
        word_positions.append([word, current_x, word_width]) 
        current_x += word_width + word_gap

reset_game()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) 
    img, faces = detector.findFaceMesh(img, draw=False)

    elapsed_time = time.time() - start_time
    remaining_time = int(game_duration - elapsed_time)

    if remaining_time <= 0:
        cv2.putText(img, "Time's up!", (500, 360), font, 2, (0, 0, 255), 4)
        cv2.putText(img, f"Final Score: {points}", (480, 420), font, 1.5, (0, 255, 255), 3)
        cv2.putText(img, "Press 'R' to restart or 'Q' to quit", (350, 480), font, 0.8, (255, 255, 255), 2)
        cv2.imshow("Blink to Select Word Game", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            reset_game()
            continue
        elif key == ord('q'):
            break
        continue

    if faces:
        face = faces[0]
        left_eye = [face[i] for i in [33, 160, 158, 133, 153, 144]]
        right_eye = [face[i] for i in [362, 385, 387, 263, 373, 380]]
        left_ratio = get_eye_aspect_ratio(left_eye)
        right_ratio = get_eye_aspect_ratio(right_eye)
        avg_ratio = (left_ratio + right_ratio) / 2
        cx, cy = face[168]
        cv2.circle(img, (cx, cy), 8, (255, 0, 255), -1)

        if avg_ratio > blink_threshold and time.time() - last_blink_time > blink_cooldown:
            last_blink_time = time.time()
            for i, (word, x, word_width) in enumerate(word_positions):
                if x - 10 < cx < x + word_width + 10 and y_position - 40 < cy < y_position + 40:
                    if word not in selected_words:
                        selected_words.append(word)
                        points += points_per_word
                        print(f"Selected: {word}")
                        if selected_words == words:
                            points += bonus_points
                            print("BONUS! All words selected in correct order!")
                        break

    speed = 2 + int(elapsed_time / 5)
    for i in range(len(word_positions)):
        word, x, word_width = word_positions[i]
        x -= speed
        if x + word_width < 0:
            max_x = max(pos[1] + pos[2] for pos in word_positions)
            x = max(max_x + word_gap, screen_width)
        word_positions[i][1] = x

    for word, x, _ in word_positions:
        color = (0, 255, 0) if word in selected_words else (255, 255, 255)
        cv2.putText(img, word, (x, y_position), font, font_scale, color, 2)

    cv2.putText(img, f"Points: {points}", (20, 50), font, 1, (0, 200, 255), 2)
    cv2.putText(img, f"Time left: {remaining_time}s", (1000, 50), font, 1, (255, 100, 100), 2)
    cv2.putText(img, "Selected: " + " ".join(selected_words), (20, 680), font, 0.8, (0, 255, 255), 2)

    cv2.imshow("Blink to Select Word Game", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        reset_game()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


