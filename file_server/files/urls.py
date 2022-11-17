from django.urls import path
from .views import  FileAPIView
from rest_framework.urlpatterns import format_suffix_patterns

urlpatterns = [

    # path('generic/user/', UserAPIView.as_view()),

    path('', FileAPIView.as_view()),
    path('download/<int:id>/', FileAPIView.as_view()),
    path('<int:pk>/', FileAPIView.as_view()),



]

# urlpatterns = format_suffix_patterns(urlpatterns)