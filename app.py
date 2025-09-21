from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import google.generativeai as genai
from llama_api_client import LlamaAPIClient

import requests
import json

API_KEY = "AIzaSyCVhZ5sv2Wha_T7q3w2tYVaJtznyYXeN0E"
genai.configure(api_key=API_KEY)

model = YOLO("yolov8n.pt")
LABELS = model.names

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

person_boxes = set()
seen_objects = defaultdict(set)

def detect_objects():
    cap = cv2.VideoCapture(0)
    start = time.time()
    duration = 10  
    print("Scanning room using unique object positions...")

    while time.time() - start < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = LABELS[cls]
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            x1r, y1r, x2r, y2r = [round(val / 20) * 20 for val in [x1, y1, x2, y2]]
            bbox_hash = (x1r, y1r, x2r - x1r, y2r - y1r)

            if label == "person":
                person_boxes.add(bbox_hash)
            else:
                seen_objects[label].add(bbox_hash)

    cap.release()
    cv2.destroyAllWindows()

    return len(person_boxes), {label: len(bboxes) for label, bboxes in seen_objects.items()}

def format_summary(person_count, other_counts):
    if not person_count and not other_counts:
        return "No objects were detected in the room."

    people_desc = ""
    if person_count == 1:
        people_desc = "There is one person in the room."
    elif person_count > 1:
        people_desc = "There are multiple people in the room."

    object_keywords = [
        "backpack", "chair", "desk", "table", "tv", "couch", "bookshelf", "locker",
        "cabinet", "door", "trash can", "recycling bin", "potted plant", "microwave",
        "refrigerator", "storage box", "crate", "ladder", "whiteboard", "chalkboard",
        "table", "phone", "water", "food"
    ]
    found_objects = [k for k in other_counts if k in object_keywords]

    object_desc = ""
    if found_objects:
        object_desc = " The following useful objects were also found: " + ", ".join(found_objects) + "."

    return people_desc + object_desc

def create_prompt(summary_text):
    return f"""
    You are LifeLine, an AI emergency assistant. The room has these objects: {summary_text}.

    Based on these objects, give exact instructions for a school lockdown. Be short, precise, and actionable:

    1. Barricade the door
    2. Best hiding spots
    3. Silencing electronics
    4. Stealth & behavior
    5. Team coordination
    6. Mental support
    """

def ask_gemini(prompt):
    try:
        client = LlamaAPIClient(
            api_key="LLM|987057120068767|ledxwaPE-QxmIHY1YHTmw87qz_o",
            base_url="https://api.llama.com/v1/",
        )
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.completion_message.content.text
    except Exception:
        return fallback_strategy()

def fallback_strategy():
    if not seen_objects:
        return "No objects detected; cannot provide lockdown strategy."

    items = list(seen_objects.keys())
    furniture = [i for i in items if i in ["desk", "chair", "table", "bookshelf", "bench"]]
    electronics = [i for i in items if i in ["tv", "computer", "laptop", "microwave"]]
    conceal_items = [i for i in furniture if i in ["bookshelf", "desk"]]

    def confidence(item):
        return round(0.7 + 0.2*(item in furniture), 2)

    output = []
    output.append("PRIMARY STRATEGY: MULTI-FURNITURE BARRICADE")

    if "desk" in furniture and "chair" in furniture:
        output.append(f"DESK + CHAIR FORTRESS (AI Confidence: {confidence('desk')})")
    if "table" in furniture and "bookshelf" in furniture:
        output.append(f"TABLE + BOOKSHELF WALL (AI Confidence: {confidence('table')})")
    if "chair" in furniture and "desk" not in furniture:
        output.append(f"CHAIR BARRICADE SYSTEM (AI Confidence: {confidence('chair')})")
    if conceal_items:
        output.append("CONCEALMENT PROTOCOL")
    if furniture:
        output.append("HIDE & MOVE")
    if electronics:
        output.append("SILENCE ELECTRONICS")
    if len(person_boxes) > 1:
        output.append("TEAMWORK")
    output.append("MENTAL SUPPORT")

    return "\n".join(output)

@app.get("/")
def read_root():
    return {"message": "LifeLine AI is running."}

@app.get("/lifeline")
def lifeline():
    person_count, other_counts = detect_objects()
    summary_text = format_summary(person_count, other_counts)
    prompt = create_prompt(summary_text)
    response = ask_gemini(prompt)
    return {
        "status": "success",
        "summary": summary_text,
        "response": response
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai:app", host="0.0.0.0", port=5002, reload=True)
