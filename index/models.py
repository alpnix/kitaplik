from django.db import models

# Create your models here.

class Kitap(models.Model): 

    title = models.CharField(max_length=2000)
    author = models.CharField(max_length=1000, default=None, blank=True, null=True)
    file = models.FileField(default=None,blank=True, null=True, upload_to="book_files/")
    text = models.TextField(max_length=9999999, default=None, blank=True, null=True)
    score = models.CharField(max_length=40, null=True, blank=True)

    class Meta:
        verbose_name = 'Kitap'
        verbose_name_plural = 'Kitaplar'

    def __str__(self): 
        return self.title

