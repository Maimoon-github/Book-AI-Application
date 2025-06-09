from django.db import models

class Document(models.Model):
    file = models.FileField(upload_to='documents/')
    name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    content = models.TextField(blank=True)

    def __str__(self):
        return self.name

class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='chunks')
    text_content = models.TextField()
    metadata = models.JSONField(null=True, blank=True)
    embedding = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Chunk {self.id} of {self.document.name}"
