from django.db import models

# Create your models here.

# Create your models here.

class File(models.Model):

    size = models.CharField(("size"), max_length=100)
    fileName = models.TextField()
    user = models.TextField()
    extention = models.CharField(("extention"), max_length=20)
    createdDate = models.CharField(("createdDate"), max_length=20)
    modifiedDate = models.CharField(("modifiedDate"), max_length=20)
    upFolder = models.TextField()
    # file = models.FileField(max_length=None, null = True)

    # date=models.DateTimeField(("date"), auto_now=False, auto_now_add=True)
    class Meta:
        db_table = 'File'

    def __str__(self):
        return self.fileName
