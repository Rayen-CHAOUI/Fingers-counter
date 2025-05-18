import customtkinter as ctk
import threading
from datetime import datetime
import cv2
from hand_counter import HandCounter

ctk.set_appearance_mode("System")  # "Dark" or "Light"
ctk.set_default_color_theme("blue")  # Can also be "dark-blue", "green", etc.

class FingerCounterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Rayen's Finger Counter")
        self.geometry("700x450")
        self.resizable(False, False)

        # Layout
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=10)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.main_content = ctk.CTkFrame(self, corner_radius=10)
        self.main_content.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        # Sidebar Buttons
        self.title_label = ctk.CTkLabel(self.sidebar, text="Finger Counter Menu", font=ctk.CTkFont(size=18, weight="bold"))
        self.title_label.pack(pady=20)

        self.start_button = ctk.CTkButton(self.sidebar, text="Start Live Finger Counter", command=self.start_finger_counter)
        self.start_button.pack(pady=10)

        self.toggle_theme_button = ctk.CTkButton(self.sidebar, text="Toggle Theme", command=self.toggle_theme)
        self.toggle_theme_button.pack(pady=10)

        self.quit_button = ctk.CTkButton(self.sidebar, text="Exit", fg_color="red", command=self.destroy)
        self.quit_button.pack(pady=10)

        # Main Content
        self.heading_label = ctk.CTkLabel(self.main_content, text="Welcome to Rayen's Finger Counter!", font=ctk.CTkFont(size=20, weight="bold"))
        self.heading_label.pack(pady=20)

        self.info_label = ctk.CTkLabel(self.main_content, text="Click the button to start detecting hand fingers\nusing your webcam.", justify="center")
        self.info_label.pack(pady=10)

        self.status_label = ctk.CTkLabel(self.main_content, text="Status: Idle", text_color="green")
        self.status_label.pack(pady=10)

        self.clock_label = ctk.CTkLabel(self.main_content, text="", font=ctk.CTkFont(size=14))
        self.clock_label.pack(pady=10)
        self.update_clock()

    def start_finger_counter(self):
        self.status_label.configure(text="Status: Running...", text_color="orange")
        thread = threading.Thread(target=self.run_counter)
        thread.start()

    def run_counter(self):
        cap = cv2.VideoCapture(0)
        detector = HandCounter()

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame, hands = detector.process(frame)

            for label, landmarks in hands:
                fingers = detector.count_fingers(label, landmarks)
                pos_x = int(landmarks[0][0] * frame.shape[1])
                pos_y = int(landmarks[0][1] * frame.shape[0])
                cv2.putText(frame, f'{label} Hand: {fingers}', (pos_x - 50, pos_y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Finger Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.status_label.configure(text="Status: Idle", text_color="green")

    def toggle_theme(self):
        current = ctk.get_appearance_mode()
        new_theme = "Light" if current == "Dark" else "Dark"
        ctk.set_appearance_mode(new_theme)

    def update_clock(self):
        now = datetime.now().strftime("%A, %d %B %Y\n%H:%M:%S")
        self.clock_label.configure(text=now)
        self.after(1000, self.update_clock)

if __name__ == "__main__":
    app = FingerCounterApp()
    app.mainloop()
