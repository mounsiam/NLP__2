# models.py

from django.db import models

class Complaint(models.Model):
    text = models.TextField(blank=True, null=True)
    audio = models.BinaryField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Complaint {self.id}"
