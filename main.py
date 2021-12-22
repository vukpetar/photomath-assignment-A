import os
import numpy as np
import cv2
from flask import Flask, jsonify, request
from utils import HandwrittenCharacterDetector, HandwrittenCharacterClassifier, Solver

app = Flask(__name__)

model_path = "./model"

@app.route('/', methods=['POST'])
def extract_entities():
    
    result = []
    if request.files['file'] is None:
        return jsonify(code=403, message="bad request")
    try:
        nparr = np.fromstring(request.files['file'].read(), np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        hcd = HandwrittenCharacterDetector(img)
        cropped_images = hcd.getCroppedImages()

        expression = ''
        for cropped_image in cropped_images:
            hcc = HandwrittenCharacterClassifier(cropped_image['img']/255)
            character, confidence = hcc.getLabel()
            expression += character

        solution = Solver.getFinalResults(expression)
        result = {
            "expression": expression,
            "solution": solution

        }
    except Exception as e:
        result.append(str(e))
    return jsonify(result)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google Cloud
    # Run, a webserver process such as Gunicorn will serve the app.
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))