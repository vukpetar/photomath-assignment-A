import os
import numpy as np
import cv2
from flask import Flask, jsonify, request
from utils import getResult

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

        solution, expression, exception = getResult(img)
        result = {
            "expression": expression,
        }
        if solution:
            result["solution"] = solution
        if exception:
            result["exception"] = exception

    except Exception as e:
        result.append(str(e))
    return jsonify(result)


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google Cloud
    # Run, a webserver process such as Gunicorn will serve the app.
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))