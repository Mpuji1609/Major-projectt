import os
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import librosa
from pydub import AudioSegment
import io
import warnings
from datetime import datetime
import joblib

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///autism_detection.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

ALLOWED_AUDIO = {'m4a', 'wav', 'mp3', 'ogg'}
ALLOWED_IMAGE = {'jpg', 'jpeg', 'png', 'bmp'}

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_type = db.Column(db.String(10), nullable=False)  # 'image' or 'audio'
    filename = db.Column(db.String(200), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Load models
try:
    image_model = load_model('best.h5')
except:
    image_model = None
    print("Warning: Image model not found")

audio_models = {
    'rf': 'Random Forest (~90% accuracy)',
    'ann': 'Artificial Neural Network (~72% accuracy)',
    'svm': 'Support Vector Machine (~54% accuracy)',
    'nb': 'Naive Bayes (~81% accuracy)',
}

# Load audio models (create dummy if not exist for demo)
loaded_audio_models = {}
for model_key, model_name in audio_models.items():
    try:
        loaded_audio_models[model_key] = joblib.load(f'{model_key}.pkl')
    except:
        loaded_audio_models[model_key] = None

# Helper functions
def allowed_file(filename, file_type):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in (ALLOWED_IMAGE if file_type == 'image' else ALLOWED_AUDIO)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def extract_audio_features(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        samples = audio.get_array_of_samples()
        y = np.array(samples).astype(np.float32) / (2**15 - 1)
        sr = audio.frame_rate
        mfcc_features = librosa.feature.mfcc(y=y, sr=sr)
        
        if not np.isnan(mfcc_features).any():
            mfcc_avg = np.mean(mfcc_features, axis=1, keepdims=True)
            return mfcc_avg.reshape(1, 20)
        return None
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def predict_image(img_path):
    if image_model is None:
        return None, None
    
    img_array = preprocess_image(img_path)
    prediction = image_model.predict(img_array)
    confidence = float(prediction[0][0])
    
    if confidence > 0.015:
        result = 'Non Autistic'
    else:
        result = 'Autistic'
    
    return result, confidence

def predict_audio(audio_path, model_key='rf'):
    if model_key not in loaded_audio_models or loaded_audio_models[model_key] is None:
        return None, None
    
    features = extract_audio_features(audio_path)
    if features is None:
        return None, None
    
    model = loaded_audio_models[model_key]
    prediction = model.predict(features)
    
    # Get confidence (mock confidence for demo)
    confidence = 0.85
    
    if prediction[0] == 1:
        result = 'Autistic'
    else:
        result = 'Non Autistic'
    
    return result, confidence

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        user = User.query.filter_by(email=data['email']).first()
        
        if user and check_password_hash(user.password, data['password']):
            session['user_id'] = user.id
            session['username'] = user.username
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        
        return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'success': False, 'message': 'Email already exists'}), 400
        
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'success': False, 'message': 'Username already exists'}), 400
        
        new_user = User(
            username=data['username'],
            email=data['email'],
            password=generate_password_hash(data['password'])
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        session['user_id'] = new_user.id
        session['username'] = new_user.username
        
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    predictions = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).all()
    
    return render_template('dashboard.html', user=user, predictions=predictions, audio_models=audio_models)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not logged in'}), 401
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    
    file = request.files['file']
    input_type = request.form.get('type')
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    if input_type == 'image' and allowed_file(file.filename, 'image'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        result, confidence = predict_image(filepath)
        
        if result:
            prediction = Prediction(
                user_id=session['user_id'],
                input_type='image',
                filename=filename,
                result=result,
                confidence=confidence
            )
            db.session.add(prediction)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'result': result,
                'confidence': float(confidence),
                'type': 'image'
            })
    
    elif input_type == 'audio' and allowed_file(file.filename, 'audio'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        
        model_key = request.form.get('model', 'rf')
        result, confidence = predict_audio(filepath, model_key)
        
        if result:
            prediction = Prediction(
                user_id=session['user_id'],
                input_type='audio',
                filename=filename,
                result=result,
                confidence=confidence
            )
            db.session.add(prediction)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'result': result,
                'confidence': float(confidence),
                'type': 'audio'
            })
    
    return jsonify({'success': False, 'message': 'Invalid file type'}), 400

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/doctors')
def doctors():
    doctors_list = [
        {
            'id': 1,
            'name': 'Dr. Sarah Johnson',
            'specialty': 'Developmental Pediatrics',
            'experience': '15 years',
            'rating': 4.8,
            'patients': 500,
            'bio': 'Specializes in early autism diagnosis and intervention programs.'
        },
        {
            'id': 2,
            'name': 'Dr. Michael Chen',
            'specialty': 'Neurodevelopmental Disorders',
            'experience': '12 years',
            'rating': 4.9,
            'patients': 450,
            'bio': 'Expert in behavioral analysis and autism spectrum assessment.'
        },
        {
            'id': 3,
            'name': 'Dr. Emma Williams',
            'specialty': 'Child Psychology',
            'experience': '10 years',
            'rating': 4.7,
            'patients': 350,
            'bio': 'Provides comprehensive evaluation and treatment planning for ASD.'
        }
    ]
    return render_template('doctors.html', doctors=doctors_list)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)