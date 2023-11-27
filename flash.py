# app.py
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import all
from config import Config

app = Flask(__name__)
app.config.from_object('config.Config')  # 使用配置文件

@app.route("/")
def hello():
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
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # 在这里可以使用 img 进行进一步的处理，例如应用你的机器学习模型

        return jsonify({"message": "Image received successfully", "result": all.start(img)})
    except Exception as e:
        # 记录异常到日志
        app.logger.error(str(e))
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == "__main__":
    # 使用 Gunicorn 启动应用程序
    app.run(host='0.0.0.0', port=5001, debug=False)