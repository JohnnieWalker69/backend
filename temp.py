from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from predicter import driver

app = Flask(__name__)
CORS(app)
current_dir = os.getcwd()

# Name of the folder you want to create a relative path for
folder_name = 'images'

# Create the relative path by joining the current directory and folder name
relative_path = os.path.join(current_dir, folder_name)
# app.config["UPLOAD_FOLDER"] = r"..\images"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in app.config["ALLOWED_EXTENSIONS"]

@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No file selected"})
    
    file = request.files["image"]
    # print(file)
    
    if file.filename == "":
        return jsonify({"error": "No file selected"})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(relative_path, filename)
        file.save(filepath)
        data = driver(filepath)
        print(data[2])
        return jsonify({"file_path": data[0],"result":data[1],"imageData":data[2]})
    else:
        return jsonify({"error": "Invalid file type"})

if __name__ == '__main__':
    app.run()