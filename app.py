from flask import Flask, render_template, request
from ultralytics import YOLO
import cv2
import os
import uuid
import pytesseract

# Path to tesseract executable (change if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
PLATES_FOLDER = os.path.join(STATIC_FOLDER, 'plates')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLATES_FOLDER, exist_ok=True)

model = YOLO('best.pt')  # Detects cars + number plates

@app.route('/')
def index():
    return render_template('index.html', full_image=None, cropped_plates=[])

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    filename = f"{uuid.uuid4()}.jpg"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    img = cv2.imread(img_path)
    results = model(img_path)[0]

    cropped_info = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names.get(cls_id, 'unknown').lower()

        if class_name != 'number_plate':
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        area = (x2 - x1) * (y2 - y1)

        if area > 100000:
            continue  # Skip large boxes (likely car)

        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Crop number plate
        cropped = img[y1:y2, x1:x2].copy()
        crop_name = f"{uuid.uuid4()}.jpg"
        crop_path = os.path.join(PLATES_FOLDER, crop_name)
        cv2.imwrite(crop_path, cropped)

        # OCR
        text = pytesseract.image_to_string(cropped, config='--psm 7')
        cropped_info.append({
            'image': f"/static/plates/{crop_name}",
            'text': text.strip()
        })

    full_result_path = os.path.join(STATIC_FOLDER, 'full_result.jpg')
    cv2.imwrite(full_result_path, img)

    return render_template(
        'index.html',
        full_image="/static/full_result.jpg",
        cropped_plates=cropped_info
    )

if __name__ == '__main__':
    app.run(debug=True)
