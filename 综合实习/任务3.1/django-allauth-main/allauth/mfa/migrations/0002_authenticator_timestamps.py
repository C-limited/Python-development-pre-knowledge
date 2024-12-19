# Generated by Django 3.2.22 on 2023-11-06 12:04

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("mfa", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="authenticator",
            name="created_at",
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AddField(
            model_name="authenticator",
            name="last_used_at",
            field=models.DateTimeField(null=True),
        ),
    ]
