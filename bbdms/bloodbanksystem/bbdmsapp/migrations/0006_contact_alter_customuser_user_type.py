# Generated by Django 5.0.7 on 2024-07-19 11:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('bbdmsapp', '0005_bloodrequest'),
    ]

    operations = [
        migrations.CreateModel(
            name='Contact',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fullname', models.CharField(max_length=250)),
                ('mobno', models.CharField(max_length=11)),
                ('email', models.EmailField(max_length=250)),
                ('message', models.TextField(max_length=250)),
                ('status', models.CharField(max_length=50)),
                ('regdate_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.AlterField(
            model_name='customuser',
            name='user_type',
            field=models.CharField(choices=[(2, 'donor'), (3, 'requester'), (1, 'admin')], default=1, max_length=50),
        ),
    ]
