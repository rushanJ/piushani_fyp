
from .models import File
from .serializers import FileSerializer,FileCreateSerializer
from rest_framework import generics
from rest_framework import mixins
import os
from django.http import HttpResponse
from django_filters.rest_framework import DjangoFilterBackend
from loguru import logger
import json
from rest_framework import status
from dotenv import load_dotenv
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import requests
import cv2 as cv
import math
import time
load_dotenv()

def returnAuthErr():
    x = {"status": "Error","message": "Authentication Error!"}
    json_format = json.dumps(x)
    # return queryset
    return HttpResponse(json_format, content_type="application/json",
                        status=status.HTTP_401_UNAUTHORIZED)
# Create your views here.
def authUser(request):
    try:
        # print(request.headers["access-token"])
        response = requests.get(os.getenv("AUTH_BASE") + "/auth",
                                 headers={"access-token":request.headers["access-token"]})
        # response = requests.get("https://api.bixchat.xyz/auth/validate", headers=request.headers)
        print(request.headers["access-token"])
        print(response.status_code)
        print(response)
        if (response.status_code == 401):
            return False, "0"
        else:
            return True, json.loads(response.text)["id"]
    except Exception as e:
        print("Error")
        print("Exception: " + str(e))
        print("Auth URL: " + str(os.getenv("AUTH_BASE")))
        # if e=="access-token":
        # x = {"status": "Error", "message": "access-token Required !"}
        # json_format = json.dumps(x)
        # return HttpResponse(json_format, content_type="application/json",
        #                     status=status.HTTP_404_NOT_FOUND)
        # logException(e, "Auth error ")
        return False, "0"


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


def age_gender_detector(frame):
    # Read frame
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame)
    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        return gender

    return frameFace

faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"

ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"

genderProto = "modelNweight/gender_deploy.prototxt"
genderModel = "modelNweight/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

padding = 20

key="omjyHOZcOWtIFX54KpI64-6Ow3M-RAsWsQWb6gpf3A8="
class FileAPIView(generics.GenericAPIView, mixins.ListModelMixin, mixins.CreateModelMixin, mixins.UpdateModelMixin,
                  mixins.DestroyModelMixin, mixins.RetrieveModelMixin):
    queryset = File.objects.all()
    serializer_class = FileCreateSerializer



    def get(self, request, *args, **kwargs):
        try:
            f = open("data.txt", "r")
            data=f.readline()
            f.close()
            x = {"data": {"status": data}}
            if (data=="1"):
                print("TRUE FOUND")
                f = open("data.txt", "w")
                f.write(str(0))
                f.close()
            json_format = json.dumps(x)
            return HttpResponse(json_format, content_type="application/json", status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(e)
            x = {"status": "Error", "message": "File Not Created !"}
            json_format = json.dumps(x)
            return HttpResponse(json_format, content_type="application/json",status=status.HTTP_500_INTERNAL_SERVER_ERROR)


    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)


