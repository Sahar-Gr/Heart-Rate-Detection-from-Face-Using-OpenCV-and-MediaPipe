import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os

class HeartRateDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.green_signal = []
        self.times = []
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.start_time = time.time()
        
        self.fig, self.ax = plt.subplots()
        self.time_data, self.signal_data = [], []
        
        # Chemin absolu du fichier CSV
        self.csv_file = "C:/Users/Sahar/Desktop/Rythme cardiaque/heart_rate_data.csv"
        
        # Écrire l'en-tête pour le fichier CSV
        try:
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Time (s)', 'Signal (Green)', 'BPM'])
            print(f"Le fichier CSV {self.csv_file} a été créé avec succès.")
        except Exception as e:
            print(f"Erreur lors de la création du fichier CSV : {e}")

    def butter_bandpass(self, lowcut, highcut, fs, order=3):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def filter_signal(self, data, lowcut=0.8, highcut=3.0):
        fs = self.fps
        b, a = self.butter_bandpass(lowcut, highcut, fs)
        return filtfilt(b, a, data)

    def process_frame(self, image, detection):
        height, width, _ = image.shape
        bbox = detection.location_data.relative_bounding_box
        x, y, w, h = int(bbox.xmin * width), int(bbox.ymin * height), int(bbox.width * width), int(bbox.height * height)

        forehead_y = y + int(0.1 * h)
        forehead_h = int(0.15 * h)
        forehead_region = image[forehead_y:forehead_y + forehead_h, x:x + w]

        if forehead_region.size != 0:
            avg_green = np.mean(forehead_region[:, :, 1])
            self.green_signal.append(avg_green)
            self.times.append(time.time() - self.start_time)

            max_samples = int(self.fps * 5)
            if len(self.green_signal) > max_samples:
                self.green_signal.pop(0)
                self.times.pop(0)

    def calculate_heart_rate(self):
        if len(self.green_signal) < 30:
            return 0

        filtered_signal = self.filter_signal(self.green_signal)

        fft = np.fft.fft(filtered_signal)
        freqs = np.fft.fftfreq(len(filtered_signal), d=1 / self.fps)
        positive_freqs = freqs[:len(freqs) // 2]  # Correction ici : remplacer 'frequencies' par 'freqs'
        magnitudes = np.abs(fft[:len(fft) // 2])

        valid_indices = (positive_freqs >= 0.8) & (positive_freqs <= 3.0)
        if not np.any(valid_indices):
            return 0

        peak_freq = positive_freqs[valid_indices][np.argmax(magnitudes[valid_indices])]
        bpm = int(peak_freq * 60)
        return bpm

    def update_graph(self):
        self.ax.clear()
        self.ax.plot(self.times, self.green_signal, color='green')
        self.ax.set_title("Signal extrait (canal vert)")
        self.ax.set_xlabel("Temps (s)")
        self.ax.set_ylabel("Amplitude")
        plt.pause(0.01)

    def detect_heart_rate(self):
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_detection.process(frame_rgb)

                if results.detections:
                    for detection in results.detections:
                        self.mp_drawing.draw_detection(frame, detection)
                        self.process_frame(frame, detection)

                bpm = self.calculate_heart_rate()

                # Enregistrement dans le CSV
                try:
                    with open(self.csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([self.times[-1], self.green_signal[-1], bpm])
                    print(f"Données enregistrées dans {self.csv_file}: {self.times[-1]}, {self.green_signal[-1]}, {bpm} BPM")
                except Exception as e:
                    print(f"Erreur lors de l'enregistrement des données dans le CSV : {e}")

                # Affichage
                cv2.putText(frame, f"BPM: {bpm}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Heart Rate Detection", frame)
                self.update_graph()

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            plt.ioff()
            plt.close()

if __name__ == "__main__":
    detector = HeartRateDetector()
    detector.detect_heart_rate()
