import cv2
import mediapipe as mp

class HandCounter:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        self.hands_module = mp.solutions.hands
        self.hands = self.hands_module.Hands(
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.drawer = mp.solutions.drawing_utils

        # Tip landmarks for each finger (thumb to pinky)
        self.finger_tips = [4, 8, 12, 16, 20]

    def process(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        hand_data = []

        if result.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = hand_info.classification[0].label  # 'Left' or 'Right'
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                hand_data.append((hand_label, landmarks))
                self.drawer.draw_landmarks(image, hand_landmarks, self.hands_module.HAND_CONNECTIONS)

        return image, hand_data

    def count_fingers(self, hand_label, landmarks):
        fingers_up = 0

        # Convert normalized coordinates to relative positions
        if landmarks:
            # Thumb
            if hand_label == "Right":
                if landmarks[4][0] > landmarks[3][0]:  # right hand, thumb to right
                    fingers_up += 1
            else:
                if landmarks[4][0] < landmarks[3][0]:  # left hand, thumb to left
                    fingers_up += 1

            # Other fingers: if tip is above PIP joint
            for tip_id in [8, 12, 16, 20]:
                if landmarks[tip_id][1] < landmarks[tip_id - 2][1]:
                    fingers_up += 1

        return fingers_up
