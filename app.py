from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from base64 import b64encode
import requests

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secret key for flash messages

class ImageProcessingApp:
    def __init__(self):
        self.image_path = None
        self.threshold = 0.70

    def process_image(self):
        if not self.image_path:
            flash("Please select an image or provide a URL.", 'error')
            return None, None

        print("Processing image path:", self.image_path)  # Add this line for debugging

        # Load the source image
        source_path = r"P_80204011000000927.tiff"
        source_image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        if source_image is None:
            flash(f"Unable to read the source image {source_path}", 'error')
            return None, None

        # Load the target image
        try:
            if self.image_path.startswith(('http://', 'https://')):
                response = requests.get(self.image_path)
                uploaded_image = Image.open(BytesIO(response.content))
                # Convert the image to grayscale using OpenCV
                target_image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2GRAY)
            else:
                target_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            flash(f"Error loading the image: {e}", 'error')
            return None, None

        print("Loaded target image")  # Add this line for debugging

        # Resize the target image to match the dimensions of the source image
        target_image = cv2.resize(target_image, (source_image.shape[1], source_image.shape[0]))

        # Ensure both images have the same depth and type
        target_image = cv2.convertScaleAbs(target_image)

        # Load all template images in the specified folder
        templates_folder = r"Logos"
        template_files = [f for f in os.listdir(templates_folder) if f.endswith((".jpg", ".jpeg", ".png", ".tiff"))]

        # Set the threshold for template matching
        threshold = self.threshold

        # Variable to check if any template is detected
        template_detected = False

        # Process each template
        for template_file in template_files:
            template_path = os.path.join(templates_folder, template_file)
            template_name = os.path.splitext(template_file)[0].split("_")[0]

            # Load the template
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

            # Ensure both template and target image have the same depth and type
            template = cv2.convertScaleAbs(template)

            # Apply template matching
            result = cv2.matchTemplate(target_image, template, cv2.TM_CCORR)
            locations = np.where(result >= threshold)
            locations = list(zip(*locations[::-1]))

            # If any match is found, set template_detected to True and break the loop
            if locations:
                template_detected = True
                break

        # If no template is detected, display a message without showing the image
        if not template_detected:
            flash("Previous report:-Logo not detected", 'info')
            return target_image, None

        # Draw rectangles around the detected areas
        for loc in locations:
            x, y = loc
            h, w = template.shape[:2]
            cv2.rectangle(target_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        print("Template detected:", template_name)  # Add this line for debugging

        return target_image, template_name

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            processing_app = ImageProcessingApp()

            uploaded_image = request.files.get('image')
            image_url = request.form.get('image_url', '').strip()

            print("Request Form Data:", request.form)
            print("Uploaded Image:", uploaded_image)
            print("Image URL:", image_url)

            if not (uploaded_image or image_url):
                flash("Please select a file or provide a URL", 'warning')
            else:
                if uploaded_image:
                    if uploaded_image.filename == '':
                        flash("No file selected", 'error')
                    else:
                        temp_image_folder = os.path.join(os.getcwd(), "temp_image")
                        if not os.path.exists(temp_image_folder):
                            os.makedirs(temp_image_folder)

                        uploaded_image_path = os.path.join(temp_image_folder, "temp_image.jpg")
                        uploaded_image.save(uploaded_image_path)

                        print("Saved uploaded image to:", uploaded_image_path)

                        processing_app.image_path = uploaded_image_path
                elif image_url:
                    processing_app.image_path = image_url

                print("Processing image path:", processing_app.image_path)

                processed_image, output_text = processing_app.process_image()
                print("Processed Image:", processed_image)
                print("Output Text:", output_text)

                if processed_image is not None:
                    _, buffer = cv2.imencode('.jpg', processed_image)
                    img_str = b64encode(buffer).decode('utf-8')
                    img_data = f'data:image/jpeg;base64,{img_str}'
                    return render_template('result.html', img_data=img_data, output_text=output_text)

                print("Processing failed. Image path:", processing_app.image_path)

        elif request.method == 'GET':
            processing_app = ImageProcessingApp()
            processing_app.image_path = None
            flash("Restarted. Please select an image or provide a URL.", 'info')

        return render_template('index.html')

if __name__ == '__main__':
    app.run(ssl_context='adhoc', debug=True)


