
# from .models import File
# from .serializers import FileSerializer,FileCreateSerializer
from rest_framework import generics
from rest_framework import mixins
from django.http import HttpResponse
from loguru import logger
import json
from rest_framework import status
from dotenv import load_dotenv

load_dotenv()


import joblib
import tensorflow as tf


def predictNN(age,area,total_weight, no_of_birds,av_weight,avfeed,fcr):
    new_model = tf.keras.models.load_model('content/nn')
    Y_pred = new_model.predict([[age,area,total_weight, no_of_birds,av_weight,avfeed,fcr]])
    # print(f"NN:{Y_pred}")
    return Y_pred

# nnRes=predictNN(4, 4, 7.5, 5, 1.51, 1.45, 0.945)

def predictRF(age,area,total_weight, no_of_birds,av_weight,avfeed,fcr):
    RF_model = joblib.load("./random_forest.joblib")
    Y_pred_rf = RF_model.predict([[age,area,total_weight, no_of_birds,av_weight,avfeed,fcr]])
    # print(f"RF:{Y_pred_rf}")
    return Y_pred_rf

# frRes=predictRF(4, 4, 7.5, 5, 1.51, 1.45, 0.945)

def predictLR(age,area,total_weight, no_of_birds,av_weight,avfeed,fcr):
    LR_model = joblib.load("./logistic_regression.joblib")
    Y_pred_lr = LR_model.predict([[age,area,total_weight, no_of_birds,av_weight,avfeed,fcr]])
    # print(f"LR:{Y_pred_lr}")
    return Y_pred_lr



def returnAuthErr():
    x = {"status": "Error","message": "Authentication Error!"}
    json_format = json.dumps(x)
    # return queryset
    return HttpResponse(json_format, content_type="application/json",
                        status=status.HTTP_401_UNAUTHORIZED)
# Create your views here.


class FileAPIView(generics.GenericAPIView, mixins.ListModelMixin, mixins.CreateModelMixin, mixins.UpdateModelMixin,
                  mixins.DestroyModelMixin, mixins.RetrieveModelMixin):
    # queryset = File.objects.all()
    # serializer_class = FileCreateSerializer



    def post(self, request, *args, **kwargs):
        try:
            age= float(request.data['age'])
            area= float(request.data['area'])
            total_weight= float(request.data['total_weight'])
            no_of_birds= float(request.data['no_of_birds'])
            av_weight= total_weight/no_of_birds
            avfeed= float(request.data['avfeed'])
            fcr= avfeed/av_weight

            print(age,area,total_weight, no_of_birds,av_weight,avfeed,fcr)
            lrRes = predictLR(age,area,total_weight, no_of_birds,av_weight,avfeed,fcr)
            frRes = predictRF(age,area,total_weight, no_of_birds,av_weight,avfeed,fcr)
            nnRes = predictNN(age, area, total_weight, no_of_birds, av_weight, avfeed, fcr)


            finalRes = (nnRes[0] + frRes[0] + lrRes[0]) / 3
            print("Final Result : ", finalRes)
            x = {"status": "Success", "message":str( round(finalRes[0]))}
            json_format = json.dumps(x)
            return HttpResponse(json_format, content_type="application/json",)


        except Exception as e:
            logger.error(e)
            x = {"status": "Error", "message": "SOMETHING  WENT WRANG !"}
            json_format = json.dumps(x)
            return HttpResponse(json_format, content_type="application/json",status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    def get(self, request, *args, **kwargs):
        try:
            age= float(request.query_params.get("age"))
            area= float(request.query_params.get("area"))
            total_weight= float(request.query_params.get("total_weight"))
            no_of_birds= float(request.query_params.get("no_of_birds"))
            av_weight= total_weight/no_of_birds
            avfeed= float(request.query_params.get("avfeed"))
            fcr= av_weight/avfeed

            print(age,area,total_weight, no_of_birds,av_weight,avfeed,fcr)
            lrRes = predictLR(age,area,total_weight, no_of_birds,av_weight,avfeed,fcr)
            frRes = predictRF(age,area,total_weight, no_of_birds,av_weight,avfeed,fcr)
            nnRes = predictNN(age, area, total_weight, no_of_birds, av_weight, avfeed, fcr)


            finalRes = (nnRes[0] + frRes[0] + lrRes[0]) / 3
            print("Final Result : ", finalRes)
            x = {"status": "Success", "message":str( round(finalRes[0]))}
            json_format = json.dumps(x)
            return HttpResponse(json_format)


        except Exception as e:
            logger.error(e)
            x = {"status": "Error", "message": "SOMETHING  WENT WRANG !"}
            json_format = json.dumps(x)
            return HttpResponse(json_format, content_type="application/json",status=status.HTTP_500_INTERNAL_SERVER_ERROR)


    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)


