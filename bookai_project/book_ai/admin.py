from django.contrib import admin
from .models import Document, Chunk

@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ('name', 'uploaded_at')
    readonly_fields = ('uploaded_at',)

@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ('id', 'document')
    search_fields = ('text_content',)
