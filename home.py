import os
from flask import Flask, request, redirect, url_for, session, send_file, after_this_request, render_template
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ImageEnhance
from io import BytesIO
import cv2
from keras.preprocessing.image import img_to_array as img_to_array_keras
from keras.models import load_model
import uuid
import numpy as np
import time
import ntpath

UPLOAD_FOLDER = 'tmp/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'tif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # img = Image.open(request.files['file'].stream)
            # session['img'] = img
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            result = perform_test(filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename),
                                  stepSize=int(request.form['stepsize']), threshold=int(request.form['threshold']))
            return render_template('upload.html', display="block", result=result, file=file.filename)
    return render_template('upload.html', display="none")


@app.route("/<subfolder>/<img_name>")
def serve_img(subfolder, img_name):
    print("image name: " + img_name)
    img = Image.open('tmp/'+subfolder+'/'+img_name)
    byte_io = BytesIO()
    img.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    lastY = False
    for y in range(0, image.shape[0], stepSize):
        # y = y
        if y + stepSize > image.shape[0]:
            y = image.shape[0] - windowSize[1]
            lastY = True
        lastX = False
        for x in range(0, image.shape[1], stepSize):
            # yield the current window

            # x = x
            if x + stepSize > image.shape[1]:
                x = image.shape[1] - windowSize[0]
                lastX = True

            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            if lastX:
                break
        if lastY:
            break


def perform_test(filepath, stepSize, threshold):
    print("stepsize: " + str(stepSize))
    # load the image
    image = cv2.imread(filepath)
    if image is None:
        raise Exception("image cannot be read or does not exist")
        exit()

    orig = Image.open(filepath)

    # resize image if too large
    maxDimension = 500
    w, h = orig.size
    # if w > maxDimension or h > maxDimension:
    # ratio = min(maxDimension/w, maxDimension/h)
    # dim = (int(w*ratio), int(h*ratio))
    # image = cv2.resize(image, dim)
    # orig = orig.resize(dim)

    # perform ELA on the image
    outDir = "tmp/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)


    elaImg = orig
    filename = outDir + uuid.uuid4().hex[:6].upper()
    elaImg.save(filename, "JPEG", quality=90)
    elaImg = Image.open(filename)
    elaImg = ImageChops.difference(orig, elaImg)
    elaImg.save(filename+"_diff", "JPEG")
    extrema = elaImg.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff
    elaImg = ImageEnhance.Brightness(elaImg).enhance(scale)
    elaImg.save(filename+"_diff2", "JPEG")
    elaImg = cv2.imread(filename+"_diff")
    cv2.imwrite('tmp/ela/'+ntpath.basename(filepath), cv2.imread(filename+"_diff2"))
    os.remove(filename)

    print("[INFO] loading network...")
    model = load_model("output/models/cnn_ela_patches.model")

    category = "Authentic"

    clone = image.copy()
    clone2 = cv2.imread(filename+"_diff", -1)
    clone3 = image.copy()
    print("shape: " + str(clone2.shape))
    (winW, winH) = (128, 128)

    for (x, y, window) in sliding_window(elaImg, stepSize=stepSize, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        window = window.astype("float") / 255.0
        window = img_to_array_keras(window)
        window = np.expand_dims(window, axis=0)
        (authentic, tampered) = model.predict(window)[0]

        # build the label
        label = "Tampered" if tampered > authentic else "Authentic"
        if label == "Tampered":
            category = "Tampered"
            for xval in range(x, x+winH):
                for yval in range(y, y+winW):
                    channels = clone2[yval][xval]
                    if channels[0] > threshold or channels[1] > threshold or channels[2] > threshold:
                        clone3[yval][xval][2] = 255

        proba = tampered if tampered > authentic else authentic
        label = "{}: {:.2f}%".format(label, proba * 100)
        proba = "{:.2f}%".format(proba * 100)

        # draw the label on the image
        print("label: " + label)

        color = (0, 0, 255) if tampered > authentic else (0, 255, 0)

        cv2.rectangle(clone, (x, y), (x + winW, y + winH), color, 2)

        cv2.putText(clone, proba, (x+10, y+25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

    cv2.imwrite('tmp/results/'+ntpath.basename(filepath), clone)
    cv2.imwrite('tmp/location/'+ntpath.basename(filepath), clone3)
    os.remove(filename+"_diff")
    os.remove(filename+"_diff2")

    return category
