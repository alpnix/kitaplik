from django.forms import ModelForm
from .models import Kitap

from django.forms import ModelForm, TextInput, FileInput

class KitapForm(ModelForm):
    class Meta:
        model = Kitap
        fields = ['title', 'author', 'file']
        widgets = {
            'title': TextInput(attrs={
                'class': "form-control",
                'id': 'book_name_input',
                'style': 'max-width: 300px;',
                'placeholder': 'Kitap ismi',
                'required': True,
            }),
                'author': TextInput(attrs={
                'class': "form-control",
                'id': 'author_name_input',
                'style': 'max-width: 300px;',
                'placeholder': 'Yazar ismi',
                'required': False,
            }),
                'file': FileInput(attrs={
                'class': "form-control", 
                'id': 'book_file_input',
                'style': 'max-width: 300px;',
                'placeholder': 'Dosya',
                'required': True,
            })
        }

