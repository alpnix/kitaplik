# Generated by Django 3.2.3 on 2023-06-05 00:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('index', '0001_initial'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='kitap',
            options={'verbose_name': 'Kitap', 'verbose_name_plural': 'Kitaplar'},
        ),
        migrations.AlterField(
            model_name='kitap',
            name='file',
            field=models.FileField(blank=True, default=None, null=True, upload_to='book_files/'),
        ),
    ]
