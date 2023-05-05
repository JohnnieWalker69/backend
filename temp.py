from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from predicter import driver

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = r"C:\Users\susha\Desktop\backend\images"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No file selected"})
    
    file = request.files["image"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        data = driver(filepath)
        print(data[2])
        return jsonify({"file_path": data[0],"result":data[1],"imageData":data[2]})
    else:
        return jsonify({"error": "Invalid file type"})

if __name__ == '__main__':
    app.run()