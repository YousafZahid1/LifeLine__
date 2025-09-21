#from fastapi import FastAPI
#from fastapi.middleware.cors import CORSMiddleware
#import cv2
#import time
#import numpy as np
#from collections import defaultdict
#from ultralytics import YOLO
#import google.generativeai as genai
#import os
#from llama_api_client import LlamaAPIClient
#
#import requests
#import json
#
#API_KEY = "AIzaSyCVhZ5sv2Wha_T7q3w2tYVaJtznyYXeN0E"
#genai.configure(api_key=API_KEY)
#
## Load YOLOv8 Nano model
#model = YOLO("yolov8n.pt")
#LABELS = model.names
#
## Initialize FastAPI app
#app = FastAPI()
#
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["*"],
#    allow_credentials=True,
#    allow_methods=["*"],
#    allow_headers=["*"],
#)
#
#person_boxes = set()
#seen_objects = defaultdict(set)
#
## Detect unique objects using bbox hashing
#def detect_objects():
#    cap = cv2.VideoCapture(0)
#    start = time.time()
#    duration = 10  # seconds
#
#    print(" Scanning room using unique object positions...")
#
#    while time.time() - start < duration:
#        ret, frame = cap.read()
#        if not ret:
#            continue
#
#        results = model(frame)[0]
#
#        for box in results.boxes:
#            cls = int(box.cls[0])
#            label = LABELS[cls]
#            xyxy = box.xyxy[0].tolist()
#            x1, y1, x2, y2 = map(int, xyxy)
#
#            # Round box positions to reduce jitter
#            x1r, y1r, x2r, y2r = [round(val / 20) * 20 for val in [x1, y1, x2, y2]]
#            bbox_hash = (x1r, y1r, x2r - x1r, y2r - y1r)
#
#            if label == "person":
#                person_boxes.add(bbox_hash)
#            else:
#                seen_objects[label].add(bbox_hash)
#
#    cap.release()
#    cv2.destroyAllWindows()
#
#    return len(person_boxes), {label: len(bboxes) for label, bboxes in seen_objects.items()}
#
## Format summary with human-friendly descriptions
#def format_summary(person_count, other_counts):
#    if not person_count and not other_counts:
#        return "No objects were detected in the room."
#
#    people_desc = ""
#    if person_count == 1:
#        people_desc = "There is one person in the room."
#    elif person_count > 1:
#        people_desc = f"There are multiple people in the room."
#
#    object_keywords = [
#        "backpack", "chair", "desk", "table", "tv", "couch", "bookshelf", "locker",
#        "cabinet", "door", "trash can", "recycling bin", "potted plant", "microwave",
#        "refrigerator", "storage box", "crate", "ladder", "whiteboard", "chalkboard","table","phone","water","food"
#    ]
#    found_objects = [k for k in other_counts if k in object_keywords]
#
#    object_desc = ""
#    if found_objects:
#        object_desc = " The following useful objects were also found: " + ", ".join(found_objects) + "."
#
#    return people_desc + object_desc
#
## Gemini Prompt
#def create_prompt(summary_text):
#    return f"""
#    You are LifeLine, an AI emergency assistant. The room has these objects: {summary_text}.
#
#    Based on these objects, give exact instructions for a school lockdown. Be **short, precise, and actionable**. Structure your response like this:
#
#    1. Barricade the door:
#       - Use specific objects in the room (e.g., desks, chairs, carts, cabinets).
#       - Tell exactly how to position them to block the door fully.
#       - Mention which objects are heaviest or most effective.
#
#    2. Best hiding spots:
#       - Identify exact places to hide behind or under (behind desks, inside cabinets).
#       - Consider line-of-sight and protection from windows or doors.
#
#    3. Silencing electronics:
#       - List any electronics that could make noise or light (phones, computers, microwaves, TVs).
#       - Give precise instructions to silence, turn off, or unplug them.
#
#    4. Stealth & behavior:
#       - Tell the user exactly how to move or stay hidden (lay flat, crouch, cover with objects).
#       - Suggest specific actions to remain invisible and silent.
#
#    5. Team coordination (if others are present):
#       - Assign tasks quietly (barricading, watching doors, monitoring hallway).
#       - Suggest hand signals or eye contact for communication.
#
#    6. Mental support:
#       - Give short, practical encouragement to stay calm and focused.
#
#    ⚠️ **Do not be vague. Do not give general advice. Focus entirely on the objects in the room.** Make each step actionable so the user can follow it immediately. Dont' have dashed OUTPUT AND FIX THE WAY FOR THE OUTPUT!
#    """
#
#
## Ask Gemini or fallback
#def ask_gemini(prompt):
#    try:
#        client = LlamaAPIClient(
#            api_key="LLM|987057120068767|ledxwaPE-QxmIHY1YHTmw87qz_o",
#            base_url="https://api.llama.com/v1/",
#        )
#
#        response = client.chat.completions.create(
#            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
#            messages=[{"role": "user", "content": prompt}],
#        )
#        return response.completion_message.content.text
#
#    except Exception:
#        # Fallback system
#        return fallback_strategy()
#
## Fallback detailed lockdown strategy
#def fallback_strategy():
#    if not seen_objects:
#        return "No objects detected; cannot provide lockdown strategy."
#
#    items = list(seen_objects.keys())
#    furniture = [i for i in items if i in ["desk", "chair", "table", "bookshelf", "bench"]]
#    electronics = [i for i in items if i in ["tv", "computer", "laptop", "microwave"]]
#    conceal_items = [i for i in furniture if i in ["bookshelf", "desk"]]
#
#    def confidence(item):
#        return round(0.7 + 0.2*(item in furniture), 2)
#
#    output = []
#    output.append(" PRIMARY STRATEGY: MULTI-FURNITURE BARRICADE")
#
#    if "desk" in furniture and "chair" in furniture:
#        output.append(f"   → DESK + CHAIR FORTRESS (AI Confidence: {confidence('desk')})")
#        output.append("     • Primary: Stack desks against door")
#        output.append("     • Secondary: Wedge chairs under door handles")
#        output.append("     • Tertiary: Create 45-degree brace system")
#
#    if "table" in furniture and "bookshelf" in furniture:
#        output.append(f"   → TABLE + BOOKSHELF WALL (AI Confidence: {confidence('table')})")
#        output.append("     • Flip table vertically against door")
#        output.append("     • Push bookshelf for additional weight")
#        output.append("     • Use books as emergency projectiles")
#
#    if "chair" in furniture and "desk" not in furniture:
#        output.append(f"   → CHAIR BARRICADE SYSTEM (AI Confidence: {confidence('chair')})")
#        output.append("     • Stack chairs pyramid-style against door")
#        output.append("     • Position legs outward to catch door frame")
#        output.append("     • Use multiple chairs for distributed weight")
#
#    if conceal_items:
#        output.append("   → CONCEALMENT PROTOCOL")
#        output.append("     • Position bookshelf or desk to block sight lines")
#        output.append("     • Immediate lights out")
#        output.append("     • Cover windows with available materials")
#
#    if furniture:
#        output.append("   → HIDE & MOVE")
#        output.append("     • Crawl behind furniture to stay out of sight")
#        output.append("     • Move slowly and silently if you need to reposition")
#
#    if electronics:
#        output.append("   → SILENCE ELECTRONICS")
#        for e in electronics:
#            output.append(f"     • Turn off or unplug {e} to prevent noise or flashing light")
#
#    if len(person_boxes) > 1:
#        output.append("   → TEAMWORK")
#        output.append("     • Use hand signals or eye contact to coordinate silently")
#        output.append("     • Assign barricade, lookout, or messenger tasks")
#
#    output.append("   → MENTAL SUPPORT")
#    output.append("     • Stay calm, focus on your plan, and breathe steadily")
#
#    return "\n".join(output)
#
## Test route
#@app.get("/")
#def read_root():
#    return {"message": "LifeLine AI is running."}
#
## Main detection + strategy route
#@app.get("/lifeline")
#def lifeline():
#    person_count, other_counts = detect_objects()
#    summary_text = format_summary(person_count, other_counts)
#    prompt = create_prompt(summary_text)
#    response = ask_gemini(prompt)
#    return {
#        "status": "success",
#        "summary": summary_text,
#        "response": response
#    }
#
## Local run
#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run("ai:app", host="0.0.0.0", port=5001, reload=True)
#
#

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import cv2
import time
from collections import defaultdict
from ultralytics import YOLO
import google.generativeai as genai
from llama_api_client import LlamaAPIClient

API_KEY = "AIzaSyCVhZ5sv2Wha_T7q3w2tYVaJtznyYXeN0E"
genai.configure(api_key=API_KEY)

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")
LABELS = model.names

# Initialize FastAPI app
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

# Detect unique objects using bbox hashing
def detect_objects():
    cap = cv2.VideoCapture(0) #original was 0
    start = time.time()
    duration = 10  # seconds
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

# Format summary with human-friendly descriptions
def format_summary(person_count, other_counts):
    if not person_count and not other_counts:
        return "No objects were detected in the room."
    people_desc = "There is one person in the room." if person_count == 1 else "There are multiple people in the room."
    object_keywords = [
        "backpack", "chair", "desk", "table", "tv", "bookshelf", "locker",
        "cabinet", "door", "trash can", "recycling bin", "potted plant", "storage box", "crate", "ladder", "whiteboard", "chalkboard",
        "phone","water","food" , "refrigator" , "desk" , "couch" , "cabnet" , "whiteboard" ,"microwave"
    ]
    found_objects = [k for k in other_counts if k in object_keywords]
    object_desc = f" The following useful objects were also found: {', '.join(found_objects)}." if found_objects else ""
    return people_desc + object_desc

# Gemini prompt creation
def create_prompt(summary_text):
    return f"""
You are LifeLine, an AI emergency assistant. The room has these objects: {summary_text}.

Based on these objects, give exact instructions for a school lockdown. Be short, precise, and actionable. Structure your response like this:

1. Barricade the door:
   - Use specific objects in the room (e.g., desks, chairs, carts, cabinets).
   - Tell exactly how to position them to block the door fully.
   - Mention which objects are heaviest or most effective.

2. Best hiding spots:
   - Identify exact places to hide behind or under (behind desks, inside cabinets).
   - Consider line-of-sight and protection from windows or doors.

3. Silencing electronics:
   - List any electronics that could make noise or light (phones, computers, microwaves, TVs).
   - Give precise instructions to silence, turn off, or unplug them.

4. Stealth & behavior:
   - Tell the user exactly how to move or stay hidden (lay flat, crouch, cover with objects).
   - Suggest specific actions to remain invisible and silent.

5. Team coordination (if others are present):
   - Assign tasks quietly (barricading, watching doors, monitoring hallway).
   - Suggest hand signals or eye contact for communication.

6. Mental support:
   - Give short, practical encouragement to stay calm and focused.
"""

# Ask Gemini or fallback
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

# Fallback lockdown strategy
def fallback_strategy():
    if not seen_objects:
        return "No objects detected; cannot provide lockdown strategy."
    items = list(seen_objects.keys())
    furniture = [i for i in items if i in ["desk", "chair", "table", "bookshelf", "bench"]]
    electronics = [i for i in items if i in ["tv", "computer", "laptop", "microwave"]]
    conceal_items = [i for i in furniture if i in ["bookshelf", "desk"]]

    def confidence(item):
        return round(0.7 + 0.2*(item in furniture), 2)

    output = ["PRIMARY STRATEGY: MULTI-FURNITURE BARRICADE"]

    if "desk" in furniture and "chair" in furniture:
        output += [
            f"DESK + CHAIR FORTRESS (AI Confidence: {confidence('desk')})",
            "• Stack desks against door",
            "• Wedge chairs under door handles",
            "• Create 45-degree brace system"
        ]
    if "table" in furniture and "bookshelf" in furniture:
        output += [
            f"TABLE + BOOKSHELF WALL (AI Confidence: {confidence('table')})",
            "• Flip table vertically against door",
            "• Push bookshelf for additional weight",
            "• Use books as emergency projectiles"
        ]
    if "chair" in furniture and "desk" not in furniture:
        output += [
            f"CHAIR BARRICADE SYSTEM (AI Confidence: {confidence('chair')})",
            "• Stack chairs pyramid-style against door",
            "• Position legs outward to catch door frame",
            "• Use multiple chairs for distributed weight"
        ]
    if conceal_items:
        output += [
            "CONCEALMENT PROTOCOL",
            "• Position bookshelf or desk to block sight lines",
            "• Immediate lights out",
            "• Cover windows with available materials"
        ]
    if furniture:
        output += [
            "HIDE & MOVE",
            "• Crawl behind furniture to stay out of sight",
            "• Move slowly and silently if you need to reposition"
        ]
    if electronics:
        output += ["SILENCE ELECTRONICS"] + [f"• Turn off or unplug {e}" for e in electronics]
    if len(person_boxes) > 1:
        output += [
            "TEAMWORK",
            "• Use hand signals or eye contact to coordinate silently",
            "• Assign barricade, lookout, or messenger tasks"
        ]
    output += ["MENTAL SUPPORT", "• Stay calm, focus on your plan, and breathe steadily"]
    return "\n".join(output)

# Routes
@app.get("/")
def read_root():
    return {"message": "LifeLine AI is running."}

@app.get("/lifeline")
def lifeline():
    person_count, other_counts = detect_objects()
    summary_text = format_summary(person_count, other_counts)
    prompt = create_prompt(summary_text)
    response = ask_gemini(prompt)
    return {"status": "success", "summary": summary_text, "response": response}

# Run locally on all interfaces
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai:app", host="0.0.0.0", port=5002, reload=True)
