# Generated by Django 5.0.7 on 2024-07-19 06:12

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bbdmsapp', '0004_alter_customuser_user_type_donorreg'),
    ]

    operations = [
        migrations.CreateModel(
            name='BloodRequest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fullname', models.CharField(max_length=250)),
                ('mobno', models.CharField(max_length=11)),
                ('email', models.EmailField(max_length=250)),
                ('requirer', models.CharField(max_length=250)),
                ('message', models.TextField(max_length=250)),
                ('regdate_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('donid', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='bbdmsapp.donorreg')),
            ],
        ),
    ]
