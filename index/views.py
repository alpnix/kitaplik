from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.core.files.storage import default_storage

from .forms import KitapForm
from .models import Kitap
from .algo import get_book_data, scale_data, predict_suitability
import os
from pdftextract import XPdf
import random

# Create your views here.

def index(request): 
    kitaplar = Kitap.objects.all()
    
    positive = "btn-outline-success" 
    negative = "btn-outline-danger"

    
    if request.method == "POST":
        if "search" in request.POST:
            print("search")
            isim = request.POST["search"]   
            kitaplar = Kitap.objects.filter(title__icontains=isim)
        elif "positive" in request.POST: 
            print("positive")
            kitaplar = Kitap.objects.filter(score__gte=0.5)
            positive = "btn-success"
            negative = "btn-outline-danger"
        elif "negative" in request.POST: 
            print("negative")
            kitaplar = Kitap.objects.filter(score__lte=0.5)
            positive = "btn-outline-success"
            negative = "btn-danger"
        elif "lang" in request.POST: 
            request.session["lang"] = "en"
            return render(request, "index/eng_index.html", {"kitaplar": kitaplar, "positive": positive, "negative": negative})
        elif "dil" in request.POST: 
            request.session["lang"] = "tr"
            return render(request, "index/index.html", {"kitaplar": kitaplar, "positive": positive, "negative": negative})
        print("results::")
        print(request.POST)
    
    lang = request.session.get("lang")

    if lang == "tr": 
        return render(request, "index/index.html", {"kitaplar": kitaplar, "positive": positive, "negative": negative})
    elif lang == "en": 
        return render(request, "index/eng_index.html", {"kitaplar": kitaplar, "positive": positive, "negative": negative})


    return render(request, "index/index.html", {"kitaplar": kitaplar, "positive": positive, "negative": negative})


def ekle(request): 


    if request.method == 'POST':
        form = KitapForm(request.POST, request.FILES)

        if  "lang" in request.POST: 
            request.session["lang"] = "en"
            return render(request, "index/eng_ekle.html", {'form': form})
        elif "dil" in request.POST: 
            request.session["lang"] = "tr"
            return render(request, "index/ekle.html", {'form': form})


        if form.is_valid():
            # file is saved
            form.save()
            
            # book_file = request.FILES["file"]
            # file_name = default_storage.save(f"book_files/{book_file.name}", book_file)
            # file_url = default_storage.url(file_name).replace(" ", "").replace("-", "")

            # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # full_path=os.path.join(BASE_DIR, file_url)

            
            book_file = request.FILES["file"]
            BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path=os.path.join(BASE_DIR, book_file.name.replace(" ", "_").replace("(", "").replace(")", ""))
            print(full_path)


            try: 
                f = open(full_path, 'r', encoding="utf8")
                text = f.read()
                f.close()
            except TypeError: 
                text = XPdf(full_path).to_text()

            book_data = get_book_data(text=None, book_location=full_path)
            scaled_book_data = scale_data(book_data)
            suitability = predict_suitability(scaled_book_data)

            k = Kitap.objects.filter(title=request.POST["title"])
            k.update(text=text)

            k.update(score=suitability)

            return redirect("index:index")
    else:
        form = KitapForm()


    lang = request.session.get("lang")

    if lang == "en": 
        return render(request, "index/eng_ekle.html", {'form': form})

    return render(request, "index/ekle.html", {'form': form})

