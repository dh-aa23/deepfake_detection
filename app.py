from flask import Flask, request, jsonify, render_template
import os
from preprocessor import Preprocess
from feature_extractor import Feature_extractor
from predictor import Predictor

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'video' not in request.files:
        return jsonify({"message": "No video file uploaded."}), 400

    video = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)
    print("video saved successfully")


    model_path = "best_model.keras" 
    face_model_path = "yolov8n.pt"


    try:
        extractor = Feature_extractor()
        feature_extractor = extractor.get_extractor()

        preprocessor = Preprocess(face_model_path, video_path, feature_extractor)
        features, mask = preprocessor.prepare_single_video()
        print("processing complete")

        predictor = Predictor(model_path)
        confidence = predictor.get_prediction(features, mask)
        
        result = 'Deepfake detected' if confidence > 0.51 else 'Real video'
        message = f"{result} (confidence: {confidence.item():.2f})"
        return jsonify({"message": message})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"Error: {str(e)}"}), 500
    finally:
        # Always delete the uploaded video file
        if os.path.exists(video_path):
            os.remove(video_path)
if __name__ == '__main__':
    app.run(debug=True)
