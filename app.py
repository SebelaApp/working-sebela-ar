'''
Holds the python calculations and implementation
'''

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import io
from io import StringIO
import base64
from PIL import Image
import cv2
import numpy as np
from flask_cors import CORS, cross_origin
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from matplotlib.colors import LinearSegmentedColormap

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
from matplotlib import pyplot as plt

app = Flask(__name__)

socketio = SocketIO(app)

upperlip = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 306, 292, 415, 310, 311,
    312, 13, 82, 81, 80, 191, 78, 62, 76, 61
]
lowerlip = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 306, 292, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 78, 62, 76, 61
]
alpha = 0.25
colorText = "color index: 0 - (w)"


def get_mpl_colormap(cmap_name):
    color_list = [
        np.array([207, 117, 80]) / 300,
        np.array([219, 179, 127]) / 300,
        np.array([248, 249, 251]) / 300,
        np.array([185, 191, 205]) / 300,
        np.array([88, 104, 127]) / 300
    ]

    cmap_name = 'Kestrel'
    N_bin = 50
    cmap = LinearSegmentedColormap.from_list('mycmap', [
        np.array([60, 0, 0]) / 255,
        np.array([140, 50, 50]) / 255,
        np.array([250, 100, 100]) / 255
    ])

    sm = plt.cm.ScalarMappable(cmap=cmap)

    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 1, 3)


colors = [
    get_mpl_colormap('brg'), cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_DEEPGREEN,
    cv2.COLORMAP_INFERNO, cv2.COLORMAP_BONE, cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_PLASMA
]


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


@socketio.on('image')
@cross_origin()
def image(data_image):
    '''
    data_image: a base64 string represeting the image retreived 
        (?) from the camera
    '''
    sbuf = StringIO()
    sbuf.write(data_image[0])
    makeupIndex = data_image[1]
    b = io.BytesIO(base64.b64decode(data_image[0]))
    pimg = Image.open(b)
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    image = frame.copy()

    def createBox(img, points, scale=5, masked=False, cropped=True):
        '''
        creates a box. This isn't dependend ton having te face in the screen
        Args:
            img: a 3D matrix, representing RGB values?? honestly i have no idea
            points: 2D a matrix of 2 columns, representing x and y coordinates (?) of each point in the specified area
        '''

        # this seems to run regardless if the face is detected so idk
        if masked:
            mask = np.zeros_like(img)
            mask = cv2.fillPoly(mask, [points], (255, 255, 255))
            img = cv2.bitwise_and(img, mask)

            B, G, R = cv2.split(img)

            B = cv2.equalizeHist(B)

            G = cv2.equalizeHist(G)

            R = cv2.equalizeHist(R)

            img = cv2.merge((B, G, R))

            L, A, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))

            # Normalize L channel by dividing all pixel values with maximum pixel value

            a = (np.mean(B))

            a = int(a) + 100

            lowRed = np.array([a, a, a])

            highRed = np.array([256, 256, 256])

            masktwo = cv2.inRange(img, lowRed, highRed)

        if cropped:
            bbox = cv2.boundingRect(points)
            x, y, w, h = bbox
            imgCrop = img[y:y + h, x:x + w]
            imgCrop = cv2.resize(imgCrop, (0, 0), None, scale, scale)
            return imgCrop
        else:
            return mask, masktwo

    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        image.flags.writeable = False
        imgOriginal = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (0, 0), None, 0.5, 0.5)

        results = face_mesh.process(image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        upperlipPoints = []
        lowerlipPoints = []

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark
            for i in upperlip:
                cord = _normalized_to_pixel_coordinates(
                    face[i].x, face[i].y, imgOriginal.shape[1],
                    imgOriginal.shape[0])
                upperlipPoints.append(cord)

            for i in lowerlip:
                cord = _normalized_to_pixel_coordinates(
                    face[i].x, face[i].y, imgOriginal.shape[1],
                    imgOriginal.shape[0])
                #cv2.circle(imgOriginal, (cord), 5, (50,50,255), cv2.FILLED)
                lowerlipPoints.append(cord)

        upperlipPoints = np.array(upperlipPoints)
        lowerlipPoints = np.array(lowerlipPoints)

        upperLipMask, upperglossMap = createBox(imgOriginal,
                                                upperlipPoints,
                                                3,
                                                masked=True,
                                                cropped=False)
        lowerLipMask, lowerglossMap = createBox(imgOriginal,
                                                lowerlipPoints,
                                                3,
                                                masked=True,
                                                cropped=False)

        redImg = np.zeros(imgOriginal.shape, imgOriginal.dtype)

        redImg[:, :] = (255, 255, 255)

        redMask = cv2.bitwise_and(redImg, redImg, mask=upperglossMap)
        blueMask = cv2.bitwise_and(redImg, redImg, mask=lowerglossMap)

        cv2.addWeighted(redMask, 0.23, imgOriginal, 1, 0, imgOriginal)
        cv2.addWeighted(blueMask, 0.23, imgOriginal, 1, 0, imgOriginal)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))

        upperLipMask = cv2.morphologyEx(upperLipMask, cv2.MORPH_CLOSE, kernel,
                                        1)
        upperLipMask = cv2.GaussianBlur(upperLipMask, (15, 15),
                                        cv2.BORDER_DEFAULT)

        inverseMaskUpper = cv2.bitwise_not(upperLipMask)
        upperLipMask = upperLipMask.astype(float) / 255
        inverseMaskUpper = inverseMaskUpper.astype(float) / 255

        lowerLipMask = cv2.morphologyEx(lowerLipMask, cv2.MORPH_CLOSE, kernel,
                                        1)

        lowerLipMask = cv2.GaussianBlur(lowerLipMask, (15, 15),
                                        cv2.BORDER_DEFAULT)
        inverseMaskLower = cv2.bitwise_not(lowerLipMask)
        lowerLipMask = lowerLipMask.astype(float) / 255
        inverseMaskLower = inverseMaskLower.astype(float) / 255

        lips = cv2.applyColorMap(imgOriginal, colors[makeupIndex])

        lips = lips.astype(float) / 255
        face = imgOriginal.astype(float) / 255

        justLipsUpper = cv2.multiply(upperLipMask, lips)
        justLipsLower = cv2.multiply(lowerLipMask, lips)
        inverseMask = cv2.multiply(inverseMaskUpper, inverseMaskLower)

        justFace = cv2.multiply(inverseMask, face)

        result = justFace + justLipsUpper + justLipsLower

        font = cv2.FONT_HERSHEY_COMPLEX

        imgencode = cv2.imencode('.jpg', result * 255)[1]

        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpg;base64,'
        stringData = b64_src + stringData
        emit('response_back', stringData)
        return "this"



socketio.run(app, host='0.0.0.0', port=8080)
