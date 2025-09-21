"""
Advanced Real-time Emotion Detection System
===========================================
Features:
- Multiple model backends (OpenCV DNN, MediaPipe, Hugging Face)
- Fallback system for reliability
- Performance optimization with threading
- Enhanced UI with emotion history and confidence meters
- Screenshot and recording capabilities
- Cross-platform compatibility

Installation:
pip install opencv-python mediapipe transformers torch pillow numpy
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
import os
from datetime import datetime
import json

# Optional imports with fallbacks
MEDIAPIPE_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✓ MediaPipe loaded successfully")
except ImportError:
    print("MediaPipe not available. Install with: pip install mediapipe")

try:
    from transformers import pipeline
    import torch
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
    print("✓ Transformers loaded successfully")
except ImportError:
    print("Transformers not available. Install with: pip install transformers torch pillow")

class EmotionDetectionSystem:
    def __init__(self):
        self.emotions = {
            0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
            4: 'sad', 5: 'surprise', 6: 'neutral'
        }
        
        # Color scheme for emotions
        self.emotion_colors = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 100, 100),    # Light Blue
            'angry': (0, 0, 255),      # Red
            'surprise': (0, 255, 255), # Yellow
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 128),  # Olive
            'neutral': (200, 200, 200) # Light Gray
        }
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.emotion_history = deque(maxlen=100)
        self.confidence_threshold = 0.6
        
        # Threading for performance
        self.processing_lock = threading.Lock()
        self.latest_results = []
        
        # UI state
        self.show_confidence_bars = True
        self.show_emotion_history = True
        self.show_fps = True
        self.recording = False
        self.video_writer = None
        
        # Initialize detection systems
        self.init_detection_systems()
    
    def init_detection_systems(self):
        """Initialize all available detection systems"""
        print("Initializing emotion detection systems...")
        
        # 1. OpenCV Haar Cascade (always available)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ OpenCV Haar Cascade loaded")
        
        # 2. MediaPipe Face Detection (more accurate)
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            print("✓ MediaPipe face detection loaded")
        else:
            self.face_detection = None
        
        # 3. Hugging Face Transformer (most accurate emotions)
        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading Hugging Face emotion model (first run downloads ~350MB)...")
                device = 0 if torch.cuda.is_available() else -1
                self.emotion_classifier = pipeline(
                    "image-classification",
                    model="trpakov/vit-face-expression",
                    device=device
                )
                print("✓ Hugging Face emotion classifier loaded")
            except Exception as e:
                print(f"Failed to load Hugging Face model: {e}")
                self.emotion_classifier = None
        else:
            self.emotion_classifier = None
        
        # 4. Fallback: Simple CNN emotion model
        self.init_fallback_model()
    
    def init_fallback_model(self):
        """Initialize a simple fallback emotion detection"""
        # This creates a basic emotion predictor using image statistics
        # In a real implementation, you'd load a pre-trained model
        self.fallback_emotions = ['neutral', 'happy', 'sad', 'angry', 'surprise']
        print("✓ Fallback emotion detection ready")
    
    def detect_faces_opencv(self, frame):
        """Detect faces using OpenCV Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        return [(x, y, x+w, y+h) for x, y, w, h in faces]
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe (more accurate)"""
        if not self.face_detection:
            return []
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                faces.append((x1, y1, x2, y2))
        
        return faces
    
            

    def predict_emotion_hf(self, face_image):
        """Predict emotion using Hugging Face transformer"""
        if not self.emotion_classifier:
            return self.predict_emotion_fallback(face_image)
        
        try:
            # Convert to RGB PIL image
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Get prediction
            results = self.emotion_classifier(pil_image, top_k=3)
            
            # Process results
            emotions_detected = []
            for result in results:
                label = result['label'].lower()
                confidence = result['score']
                
                # Map complex labels to simple emotions
                if 'happy' in label or 'joy' in label:
                    emotion = 'happy'
                elif 'sad' in label or 'sorrow' in label:
                    emotion = 'sad'
                elif 'angry' in label or 'anger' in label:
                    emotion = 'angry'
                elif 'surprise' in label:
                    emotion = 'surprise'
                elif 'fear' in label:
                    emotion = 'fear'
                elif 'disgust' in label:
                    emotion = 'disgust'
                else:
                    emotion = 'neutral'
                
                emotions_detected.append({
                    'emotion': emotion,
                    'confidence': confidence,
                    'raw_label': result['label']
                })
 
            
            return emotions_detected[0] if emotions_detected else {'emotion': 'neutral', 'confidence': 0.5}
            # return self.return_emotion(emotions_detected)
        except Exception as e:
            print(f"HF prediction error: {e}")
            return self.predict_emotion_fallback(face_image)
    
    def predict_emotion_fallback(self, face_image):
        """Simple fallback emotion prediction based on image statistics"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Simple heuristics (for demo purposes)
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Very basic emotion mapping (you'd replace this with a real model)
        if brightness > 120 and contrast > 30:
            emotion = 'happy'
            confidence = min(0.7, brightness / 180)
        elif brightness < 80:
            emotion = 'sad'
            confidence = min(0.6, (180 - brightness) / 180)
        elif contrast > 50:
            emotion = 'angry'
            confidence = min(0.6, contrast / 100)
        else:
            emotion = 'neutral'
            confidence = 0.5
        
        return {'emotion': emotion, 'confidence': confidence}
    
    def draw_enhanced_ui(self, frame, faces_emotions, fps):
        """Draw enhanced UI with emotion info, confidence bars, and stats"""
        height, width = frame.shape[:2]
        
        # Draw semi-transparent overlay for stats
        overlay = frame.copy()
        
        # Main emotion detection results
        for i, (face_coords, emotion_data) in enumerate(faces_emotions):
            x1, y1, x2, y2 = face_coords
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw face bounding box
            thickness = 3 if confidence > self.confidence_threshold else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw emotion label
            label = f"{emotion.upper()}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x1, y1-35), (x1+label_size[0]+10, y1), color, -1)
            cv2.putText(frame, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            
            # Confidence bar
            if self.show_confidence_bars:
                bar_width = int(150 * confidence)
                cv2.rectangle(frame, (x1, y2+5), (x1+150, y2+20), (50,50,50), -1)
                cv2.rectangle(frame, (x1, y2+5), (x1+bar_width, y2+20), color, -1)
                cv2.putText(frame, f"{confidence:.2f}", (x1+155, y2+17),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # FPS counter
        if self.show_fps:
            cv2.rectangle(overlay, (10, 10), (120, 50), (0,0,0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (15, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (width-30, 30), 15, (0,0,255), -1)
            cv2.putText(frame, "REC", (width-100, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        
        # Emotion history graph
        if self.show_emotion_history and len(self.emotion_history) > 1:
            self.draw_emotion_graph(frame)
        
        # Instructions
        instructions = [
            "Controls:",
            "Q - Quit",
            "S - Screenshot",
            "R - Record",
            "C - Toggle confidence",
            "H - Toggle history",
            "F - Toggle FPS"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width-200, height-150+i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    def draw_emotion_graph(self, frame):
        """Draw emotion history graph"""
        height, width = frame.shape[:2]
        graph_height = 80
        graph_width = 200
        start_x = width - graph_width - 20
        start_y = height - graph_height - 50
        
        # Draw graph background
        cv2.rectangle(frame, (start_x, start_y),
                     (start_x + graph_width, start_y + graph_height),
                     (30, 30, 30), -1)
        
        # Draw emotion frequency bars
        emotion_counts = {}
        recent_emotions = list(self.emotion_history)[-50:]  # Last 50 emotions
        
        for emotion_data in recent_emotions:
            emotion = emotion_data['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if emotion_counts:
            max_count = max(emotion_counts.values())
            bar_width = graph_width // len(emotion_counts)
            
            for i, (emotion, count) in enumerate(emotion_counts.items()):
                bar_height = int((count / max_count) * (graph_height - 10))
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                
                x = start_x + i * bar_width
                y = start_y + graph_height - bar_height
                
                cv2.rectangle(frame, (x, y), (x + bar_width - 2, start_y + graph_height), color, -1)
                
                # Label
                cv2.putText(frame, emotion[:3], (x, start_y + graph_height + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    
    def save_screenshot(self, frame):
        """Save screenshot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        return filename
    
    def toggle_recording(self, frame):
        """Toggle video recording"""
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_recording_{timestamp}.mp4"
            
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
            
            self.recording = True
            print(f"Started recording: {filename}")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            print("Recording stopped")
    
    def process_frame(self, frame):
        """Process frame to detect faces and emotions"""
        start_time = time.time()
        
        # Detect faces (try MediaPipe first, fallback to OpenCV)
        faces = []
        if MEDIAPIPE_AVAILABLE:
            faces = self.detect_faces_mediapipe(frame)
        
        if not faces:
            faces = self.detect_faces_opencv(frame)
        
        # Debug: Print number of faces detected
        if len(faces) > 0:
            print(f"Detected {len(faces)} face(s)")
        
        # Process each face
        faces_emotions = []
        for face_coords in faces:
            x1, y1, x2, y2 = face_coords
            
            # Extract and validate face region
            face_region = frame[max(0,y1):min(frame.shape[0],y2),
                              max(0,x1):min(frame.shape[1],x2)]
            
            if face_region.size == 0:
                continue
            
            # Predict emotion
            emotion_data = self.predict_emotion_hf(face_region)
            print(f"Emotion detected: {emotion_data['emotion']} (confidence: {emotion_data['confidence']:.2f})")
            faces_emotions.append((face_coords, emotion_data))
            
            # Update history
            emotion_data['timestamp'] = time.time()
            self.emotion_history.append(emotion_data)
        
        # Calculate FPS
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        self.fps_history.append(fps)
        avg_fps = np.mean(list(self.fps_history))
        
        return faces_emotions, avg_fps
    
    def run(self):
        """Main execution loop"""
        print("\n=== Advanced Emotion Detection System ===")
        print("Initializing camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot access camera. Please check camera permissions.")
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized. Starting emotion detection for 8 seconds...")
        print("Press 'Q' to quit early, 'S' for screenshot, 'R' to record")
        
        # Set timer for 8 seconds
        start_time = time.time()
        duration = 8.0  # 8 seconds
        
        try:
            while True:
                # Check if 8 seconds have passed
                elapsed_time = time.time() - start_time
                if elapsed_time >= duration:
                    print(f"\n8 seconds completed! Stopping emotion detection...")
                    break
                
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                faces_emotions, fps = self.process_frame(frame)
                
                # Draw UI
                self.draw_enhanced_ui(frame, faces_emotions, fps)
                
                # Add countdown timer to frame
                remaining_time = max(0, duration - elapsed_time)
                progress = elapsed_time / duration
                
                # Draw progress bar
                bar_width = 200
                bar_height = 20
                bar_x = 15
                bar_y = 80
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                # Progress bar
                progress_width = int(bar_width * progress)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
                
                # Timer text
                cv2.putText(frame, f"Time: {remaining_time:.1f}s", (15, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Progress: {progress*100:.0f}%", (bar_x + bar_width + 10, bar_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Record if enabled
                if self.recording and self.video_writer:
                    self.video_writer.write(frame)
                
                # Display frame
                cv2.imshow('Advanced Emotion Detection (3s Timer)', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    self.save_screenshot(frame)
                elif key == ord('r') or key == ord('R'):
                    self.toggle_recording(frame)
                elif key == ord('c') or key == ord('C'):
                    self.show_confidence_bars = not self.show_confidence_bars
                elif key == ord('h') or key == ord('H'):
                    self.show_emotion_history = not self.show_emotion_history
                elif key == ord('f') or key == ord('F'):
                    self.show_fps = not self.show_fps
        
        except KeyboardInterrupt:
            print("\nStopping emotion detection...")
        
        finally:
            # Cleanup
            if self.recording and self.video_writer:
                self.video_writer.release()
            cap.release()
            cv2.destroyAllWindows()
            
            # Save session stats
            self.save_session_stats()
            
            # Show final results
            print(f"\nEmotion detection completed in {elapsed_time:.1f} seconds!")
            print("Final results saved to emotion_session_stats.json")
    
    def save_session_stats(self):
        """Save session statistics"""
        if not self.emotion_history:
            return
        
        stats = {
            'session_duration': len(self.emotion_history),
            'emotions_detected': {},
            'average_confidence': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        total_confidence = 0
        for emotion_data in self.emotion_history:
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            
            stats['emotions_detected'][emotion] = stats['emotions_detected'].get(emotion, 0) + 1
            total_confidence += confidence
        
        stats['average_confidence'] = total_confidence / len(self.emotion_history)
        
        with open('emotion_session_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Session stats saved to emotion_session_stats.json")
        print(f"Emotions detected: {stats['emotions_detected']}")
        print(f"Average confidence: {stats['average_confidence']:.2f}")
    
    def get_session_stats_dict(self):
        """Get current session statistics as a dictionary"""
        if not self.emotion_history:
            return {}
        
        stats = {
            'session_duration': len(self.emotion_history),
            'emotions_detected': {},
            'average_confidence': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        total_confidence = 0
        for emotion_data in self.emotion_history:
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            
            stats['emotions_detected'][emotion] = stats['emotions_detected'].get(emotion, 0) + 1
            total_confidence += confidence
        
        if len(self.emotion_history) > 0:
            stats['average_confidence'] = total_confidence / len(self.emotion_history)
        
        return stats

def main():
    """Main function"""
    print("=== Advanced Real-time Emotion Detection ===")
    print("\nRequired packages:")
    print("pip install opencv-python mediapipe transformers torch pillow numpy")
    
    try:
        detector = EmotionDetectionSystem()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure your camera is connected and permissions are granted.")

if __name__ == "__main__":
    main()
