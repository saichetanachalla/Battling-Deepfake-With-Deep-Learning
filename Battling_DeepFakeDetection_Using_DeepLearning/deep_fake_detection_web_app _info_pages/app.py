import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import uuid
from tensorflow.keras.preprocessing import image
from PIL import Image
# Load the pre-trained model




import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import io
from PIL import Image
import cv2



import random







#loaded_model_imageNet = load_model("model_resnet50.h5")


from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.models import Model

# Rebuild the model architecture
resnet50_model = ResNet50(input_shape=(100, 100, 3), include_top=False, weights='imagenet', pooling='avg')
efficientnet_model = EfficientNetB0(input_shape=(100, 100, 3), include_top=False, weights='imagenet', pooling='avg')

# Freeze layers
resnet50_model.trainable = False
efficientnet_model.trainable = False

# Create inputs and ensemble architecture
inputs = Input(shape=(100, 100, 3))
resnet_output = resnet50_model(inputs)
efficientnet_output = efficientnet_model(inputs)
combined = Concatenate()([resnet_output, efficientnet_output])
x = Dense(128, activation='relu')(combined)
x = Dense(128, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)

loaded_model_imageNet = Model(inputs=inputs, outputs=outputs)

# Load the weights
loaded_model_imageNet.load_weights("model_resnet50_efficientnet_weights.h5")





#loaded_model_imageNet = load_model("model_resnet50.h5")
def pred_img(frame):
    # Resizing the frame (or face) to the model's expected input size
    resized_frame = cv2.resize(frame, (100, 100))
    
    # Convert the frame to an array and expand dimensions to match model input
    img_array = image.img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the input as expected by ResNet50
    img_array = preprocess_input(img_array)
    
    # Make predictions
    result = loaded_model_imageNet.predict(img_array)
    
    # Multiply by 100 to convert to percentages (if needed)
    final_list_result = (result * 100).astype('int')
    list_vals = list(final_list_result[0])
    
    # Get the highest confidence score and its index
    result_val = max(list_vals)
    index_result = list_vals.index(result_val)

    # Print or return the array of predictions, index, and confidence score
    print("The array of predictions is:", list_vals)
    print("The index value is:", index_result)
    print("The confidence score is:", result_val)

    # Return both the index of the predicted class and the confidence score
    return index_result, result_val
# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
import torch
# Load the trained model
#model = load_model('deepfake_face.h5')

# Check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# Home route
@app.route('/', methods=['GET', 'POST'])
def index():


    return render_template('index9.html')






# Home route
@app.route('/index2', methods=['GET', 'POST'])
def index2():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = str(uuid.uuid4()) + '_' + file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the video
            processed_frames = process_video(file_path)
            
            # Display the result
            return render_template('index.html', frames=processed_frames)
    
    return render_template('index.html')
# Home route
# Home route
'''@app.route('/index3', methods=['GET', 'POST'])
def index3():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # If the user does not select a file, the browser may submit an empty file
        if file.filename == '':
            return 'No selected file'
        if file:
            # Save the file as output.png
            file_path = os.path.join('output.png')
            file.save(file_path)
            rslt=process_image('output.png')
            #frames = []  # Replace with actual frame extraction logic if needed
            return render_template('index3.html', rslt=rslt)  # Pass any data needed for rendering
    return render_template('index3.html', frames=None)'''
# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/index3', methods=['GET', 'POST'])
def index3():
    uploaded_image = None
    is_fake = None

    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Save the file in the upload folder
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Generate URL for the uploaded file
            uploaded_image = url_for('static', filename=f'uploads/{file.filename}')

            # Dummy deepfake detection logic (replace this with your model's output)
            is_fake = "fake" in file.filename.lower()

    # Pass the result and uploaded image URL to the template
    return render_template('index3.html', uploaded_image=uploaded_image, is_fake=is_fake)


# Paths to the real and fake images
REAL_IMAGES_FOLDER = 'static/q_images/real'
FAKE_IMAGES_FOLDER = 'static/q_images/fake'

# Load images
def load_images():
    real_images = [os.path.join('q_images/real/', f) for f in os.listdir(REAL_IMAGES_FOLDER) if f.endswith(('png', 'jpg', 'jpeg'))]
    fake_images = [os.path.join('q_images/fake/', f) for f in os.listdir(FAKE_IMAGES_FOLDER) if f.endswith(('png', 'jpg', 'jpeg'))]
    return real_images, fake_images
'''
@app.route('/index5', methods=['GET', 'POST'])
def index5():
    if request.method == 'POST':
        # Collect responses
        responses = request.form.getlist('response')
        images = request.form.getlist('image')
        results = []

        for img, user_response in zip(images, responses):
            # Determine if the response was correct
            if 'real' in img and user_response == 'real':
                results.append((img, 'Correct'))
            elif 'fake' in img and user_response == 'fake':
                results.append((img, 'Correct'))
            else:
                results.append((img, 'Incorrect'))

        return render_template('results.html', results=results)

    # Load images and shuffle them
    real_images, fake_images = load_images()
    all_images = real_images + fake_images
    random.shuffle(all_images)
    selected_images = all_images[:10]  # Select 20 images randomly

    return render_template('index5.html', images=selected_images)'''
@app.route('/index5', methods=['GET', 'POST'])
def index5():
    if request.method == 'POST':
        # Collect user responses
        responses = request.form.getlist('response')
        images = request.form.getlist('image')
        results = []
        score = 0

        for img, user_response in zip(images, responses):
            # Determine if the response was correct
            if 'real' in img and user_response == 'real':
                results.append((img, 'Correct'))
                score += 1
            elif 'fake' in img and user_response == 'fake':
                results.append((img, 'Correct'))
                score += 1
            else:
                results.append((img, 'Incorrect'))

        # Calculate the final score
        total_questions = len(images)
        percentage = (score / total_questions) * 100

        # Determine feedback message
        if percentage == 100:
            feedback = "Outstanding! You have an excellent eye for spotting deepfakes."
        elif percentage >= 70:
            feedback = "Great job! You're pretty good at identifying deepfakes, but there's room for improvement."
        elif percentage >= 40:
            feedback = "Good effort! You might want to learn more about how deepfakes are made and detected."
        else:
            feedback = "It seems you're struggling to identify deepfakes. Don't worry, we've provided resources below to help you learn more."

        # Resources and references
        references = [
            {"title": "Understanding Deepfakes", "link": ""},
            {"title": "How to Spot Deepfake Videos", "link": ""},
            {"title": "Deepfake Detection Challenges", "link": ""},
            {"title": "AI and Ethics of Deepfakes", "link": ""},
        ]

        # Render results page with score and references
        return render_template(
            'index5.html',
            results=results,
            score=score,
            percentage=percentage,
            feedback=feedback,
            references=references,
            images=None,  # Ensure no images are shown on results page
        )

    # For GET request: Display the quiz with images
    real_images, fake_images = load_images()
    all_images = real_images + fake_images
    random.shuffle(all_images)
    selected_images = all_images[:10]  # Select 10 images randomly
    return render_template('index5.html', images=selected_images)

def process_image(image_path):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Initialize counters for prediction results
    real_count = 0
    fake_count = 0
    
    # Process each detected face
    for (x, y, w, h) in faces:
        # Crop the face region
        face = image[y:y+h, x:x+w]
        
        # Pass the face to the prediction function
        prediction_label, result_val2 = pred_img(face)
        
        # Count the predictions
        if prediction_label == 1:
            real_count += 1
        else:
            fake_count += 1
        
        # Print or log the label prediction
        print("Predicted label for the detected face:", "Real" if prediction_label == 1 else "Fake")
    
    # Final result based on majority vote or another logic
    result = "Real" if real_count > fake_count else "Fake"
    print(f"Overall prediction: {result}")
    
    # Return the overall result
    return result

def process_video(video_path):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize variables
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = np.linspace(0, frame_count-1, 20, dtype=int)
    processed_frames = []
    
    real_count = 0
    fake_count = 0
    
    # First pass: Process the selected frames and count predictions
    for i, frame_idx in enumerate(frames_to_process):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert frame to grayscale for face detection
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            continue
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Crop the face region
            face = frame[y:y+h, x:x+w]
            #resized_face = cv2.resize(face, (64, 64))
            #img_array = img_to_array(face)
            #img_array = np.expand_dims(img_array, axis=0)
            #img_array = img_array / 255.0
            
            # Make predictions
            #predictions = model.predict(img_array)
            predictions,result_val2 = pred_img(face)  # Passing the first element of the batch
           # predicted_class = np.argmax(predictions, axis=1)[0]
            
            # Count the predictions
           # if predicted_class == 0:
           #     real_count += 1
           # else:
           #     fake_count += 1
            print("the label predicted is ,",predictions)
            #predictions = pred_img(img_array[0])  # Passing the first element of the batch
            #predicted_class = torch.argmax(predictions, dim=1).item()  # Use torch.argmax

            # Count the predictions
            if predictions == 1:
                real_count += 1
            else:
                fake_count += 1    
    # Determine the majority label
    majority_label = 'Real' if real_count > fake_count else 'Fake'
    
    # Reprocess the frames with the majority label
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i, frame_idx in enumerate(frames_to_process):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            continue
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add the majority prediction label on the frame
            cv2.putText(frame, majority_label+"score="+str(result_val2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert the frame to RGB for display and encode it to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_data = base64.b64encode(buffer).decode('utf-8')
        
        # Append the processed frame and prediction
        processed_frames.append({
            'image': f"data:image/jpeg;base64,{frame_data}",
            'prediction': majority_label,
            'bbox': (x, y, w, h)
        })
    
    cap.release()
    return processed_frames




if __name__ == '__main__':
    app.run(debug=True)
