from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import all


app = Flask(__name__)

@app.route("/")
def Hello():
    return "Hello"

@app.route("/api", methods=["POST"])
def upload_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        image_data = file.read()
        base64_data = base64.b64encode(image_data) 
        with open('1.jpg', 'wb') as file:
            jiema = base64.b64decode(base64_data)   # 解码
            file.write(jiema)
        
        img_np = np.frombuffer(jiema, dtype=np.uint8)
        # print("Before decoding:", img_decode_bytes)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        # print("After decoding:", img)

        # 在這裡可以使用 img 進行進一步的處理，例如應用你的機器學習模型

        return jsonify({"message": "Image received successfully", "result" : all.start(img)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug = False)
