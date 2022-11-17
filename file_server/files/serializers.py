from rest_framework import serializers
from .models  import File


class FileSerializer(serializers.Serializer):
      class Meta:
            # model = File
            fields = ['id','upFolder','file']

class FileCreateSerializer(serializers.ModelSerializer):
      class Meta:
            model = File
            fields = ['id', 'size', 'fileName',  'extention', 'createdDate', 'modifiedDate', 'upFolder']

