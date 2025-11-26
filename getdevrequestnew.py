from flask import Flask, request, jsonify
import requests, json
import uuid
import os, gc
import glob
import threading
import numpy as np
import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import warnings
import redis
import time

with tf.device('/CPU:0'):
    pass

warnings.filterwarnings('ignore')

results_lock = threading.RLock()
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
app = Flask(__name__)


def prediction_model_overlap(image_path):
    with tf.device('/CPU:0'):
        with results_lock:
            classes = ['ECG', 'NO_SIGNAL', 'SIGNAL_OVERLAP']
            image = Image.open(image_path).convert('RGB')
            # image = cv2.imread(image_path)
            input_arr = np.array(image, dtype=np.float32)
            input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
            input_arr = tf.expand_dims(input_arr, axis=0)
            over_interpreter.set_tensor(over_input_details[0]['index'], input_arr)
            over_interpreter.invoke()
            output_data = over_interpreter.get_tensor(over_output_details[0]['index'])
            idx = np.argmax(output_data[0])
            return output_data[0], classes[idx]


def prediction_model(image_path):
    with tf.device('/CPU:0'):
        with results_lock:
            classes = ['ECG', 'No ECG']
            image = Image.open(image_path).convert('RGB')
            input_arr = np.array(image, dtype=np.float32)
            input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
            input_arr = tf.expand_dims(input_arr, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_arr)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            idx = np.argmax(output_data[0])
            return output_data[0], classes[idx]


def convert_png_to_jpeg(input_path):
    """
    Converts an AVIF or PNG image to JPEG format with .jpeg extension and overwrites the original file.
    - If input is .avif, it first converts to .png, then to .jpeg.
    - Deletes intermediate files.
    - Does nothing if the file is not a PNG or AVIF.

    Parameters:
    - input_path: str - Path to the input image file.
    """
    try:
        # Handle .avif files
        if input_path.lower().endswith('.avif') or input_path.lower().endswith('.webp'):
            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGBA")  # AVIF may include transparency
                    base = os.path.splitext(input_path)[0]
                    png_path = f"{base}.png"
                    img.save(png_path, "PNG")
                os.remove(input_path)
                print(f"Converted AVIF to PNG and removed original: {input_path}")
                input_path = png_path  # Update to continue PNG ? JPEG
            except Exception as e:
                print(f"Error converting AVIF '{input_path}': {e}")
                return input_path

        # Handle .png files (including ones that were AVIF originally)
        if not input_path.lower().endswith('.png'):
            return input_path

        with Image.open(input_path) as img:
            img = img.convert("RGB")  # JPEG doesn't support alpha
            base = os.path.splitext(input_path)[0]
            jpeg_path = f"{base}.jpeg"
            img.save(jpeg_path, "JPEG")

        os.remove(input_path)
        print(f"Converted PNG to JPEG and removed original: {input_path}")
        return jpeg_path

    except Exception as e:
        print(f"Error processing '{input_path}': {e}")
        return input_path

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload Files REQUEST...")
    data = request.json
    print("response:", data)
    file = request.files.get("image")

    if '_id' not in data or 'path' not in data or 'image' not in data:
        return jsonify({'error': 'Invalid input'}), 400

    get_response = {}
    try:
        get_response['_id'] = data['_id']
        get_response['path'] = data['path']
        get_response['image'] = data['image']
        get_response['user_id'] = data['userId']
        if "isModifyResting" in data:
            if data['isModifyResting'] == "false":
                get_response['is_modify_resting'] =  False
            else:
                get_response['is_modify_resting'] = data['isModifyResting']
        else:
            get_response['is_modify_resting'] =  False
    except Exception as e:
        print(f"Error parsing fields: {e}")
        return jsonify({'error': 'Missing required field'}), 400

    if get_response['is_modify_resting'] == False:
        redis_data = json.dumps(get_response)

        queue_key = f"user_queue:{get_response['user_id']}"
        redis_client.rpush(queue_key, redis_data)

        redis_client.zadd("user_priority_zset", {get_response['user_id']: time.time()}, nx=True)

        return jsonify({"message": "Image uploaded successfully"}), 200
    else:
        print("===========================")
        file_path = os.path.join("newimages", get_response['image'])
        print(file_path,"======")
        url = f'https://oeadev.projectkmt.com/oea/api/v1/uploads/images/{data["userId"]}/{data["image"]}'

        response = requests.get(url)
        with open(file_path, 'wb') as file:
            file.write(response.content)

        if os.path.exists(file_path):
            paper_ecg_url = "http://192.168.2.66:1200/upload-ecg/" # "http://192.168.2.236:1200/upload-ecg/"
            payload = {
                "file_id": get_response['_id'],
                "user_id": get_response['user_id']
            }
            files = {
                "file": open(file_path, "rb")
            }
            response = requests.post(paper_ecg_url, data=payload, files=files)
            print(response.json())  
            if response.status_code == 200:  
                return jsonify({"message": "Image uploaded successfully"}), 200
            else:
                return jsonify({"error": "Failed to upload image"}), 400  
        else:
            return jsonify({"error": "Failed to upload image"}), 400  
    return '', 200


@app.route('/checkImage', methods=['GET'])
def checkImage():
    print("GET REQUEST...")
    try:
      tf.keras.backend.clear_session()
      gc.collect()
      print(request.args.to_dict(flat=False))
      responses = request.args.get("image")
      userId = request.args.get("userId")
      format = request.args.get("formate")
      paperspeed = request.args.get("paperspeed")
      voltagegain = request.args.get("voltagegain")
#      print(responses)
      url = f'https://oeadev.projectkmt.com/oea/api/v1/uploads/images/{userId}/{responses}'
  
      response = requests.get(url)
#      print(response)
  
      if response.status_code == 200:
          image = f"newimages/{responses}"
          with open(image, 'wb') as file:
              file.write(response.content)
  
      image_path = f"newimages/{responses}"
      file_name = os.path.splitext(os.path.basename(image_path))[0]
#      print(f"Processing {file_name}")
  
      if image_path.lower().endswith('.avif') or image_path.lower().endswith('.webp'):
          image_path = convert_png_to_jpeg(image_path)
#      print(image_path)
  
      image = Image.open(image_path).convert('RGB')
  
      is_small_image = False
      file_size_kb = os.path.getsize(image_path) / 1024
      if file_size_kb < 50:
          print(f"Image {image_path} must be at least greater than 50KB.")
          is_small_image = True
  
      input_arr = np.array(image, dtype=np.float32)
      input_arr = tf.image.resize(input_arr, size=(224, 224), method=tf.image.ResizeMethod.BILINEAR)
#      print('check: ', input_arr.shape)
  
      if input_arr.shape[-1] != 3:
          image_path = convert_png_to_jpeg(image_path)
  
      with tf.device('/CPU:0'):
          output_data, class_name = prediction_model(image_path)
          if class_name == 'No ECG':
              class_name = 'NO_ECG'
  
      print('check is ecg: ', class_name)
  
      if class_name == "ECG":
          over_output_data, ove_class_name = prediction_model_overlap(image_path)
          print('Signal type check: ', ove_class_name)
          msg = "ECG"
          status = "success"
          base_time = 50  # Avg time per image
  
          if ove_class_name == "NO_SIGNAL":
              return {"status": "fail", "type": "NO_SIGNAL"}
  
          elif ove_class_name == "SIGNAL_OVERLAP":
              return {"status": "fail", "type": "SIGNAL_OVERLAP"}
  
          elif is_small_image:
              return {"status": "fail", "type": "NO_SIGNAL"}
  
          # Use a Redis counter to track processing position globally
          pending_key = "global_processing_counter"
          current_count = redis_client.incr(pending_key)
          redis_client.expire(pending_key, 60)  # Optional: auto-clear the counter after 60 seconds
  
          # Each image is assumed to take 50 seconds
          estimated_time = current_count * 45
  
          img_response = {
              "status": status,
              "type": class_name,
              "Message": msg,
              "time": estimated_time
          }
  
          print("Estimated Time:", estimated_time, "sec")
          print("Response: ", img_response)
  
          return img_response
      else:
          return {"status": "fail", "type":  "NO_SIGNAL"}

    except Exception as e:
        print(e)
        return {"status": "fail", "type": "NO_SIGNAL"}
        
@app.route('/uploadcsv', methods=['POST'])
def upload_csv():
#    print("**********  CVS API CALLED **********")
    file_id = request.form.get('id')
    file = request.files.get('file')
    user_id = request.form.get('user_id')
    csv_file_name = request.form.get('csv_name')
    img_name = request.form.get('img_name')

    if not file or not file_id:
        return jsonify({"error": "Missing file or id"}), 400

    os.makedirs("resting_csv_data", exist_ok=True)
    file_path = os.path.join("resting_csv_data", file.filename)

    file.save(file_path)  # Simpler than using shutil for Flask uploads
#    print(os.path.exists(file_path),"++++++++")

    if os.path.exists(file_path):
        queue_key = f"user_csv_queue:{user_id}"
#        print(queue_key,"======")
        temp_data = {
            "file_id": file_id,
            "user_id": user_id,
            "csv_file_name": csv_file_name,
            "img_name": img_name
        }
        redis_data = json.dumps(temp_data)
        redis_client.rpush(queue_key, redis_data)
        redis_client.zadd("user_priority_zset_csv", {user_id: time.time()}, nx=True)
        queue_length = redis_client.llen(queue_key)
#        print(queue_length,"=====")
        
        return jsonify({
            "message": "CSV file uploaded successfully",
            "file": file.filename,
            "file_id": file_id
        })
    else:
        return jsonify({"error": "Failed to save file"}), 500

if __name__ == '__main__':
    port = 1600
    model_path = 'Model/restingecgModel_5.tflite'
    over_model = 'Model/Restingecg_overlap_10.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    over_interpreter = tf.lite.Interpreter(model_path=over_model)
    over_interpreter.allocate_tensors()
    over_input_details = over_interpreter.get_input_details()
    over_output_details = over_interpreter.get_output_details()
    print(f"Service Active on Port:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)

