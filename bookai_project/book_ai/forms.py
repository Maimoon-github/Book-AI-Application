from django import forms
from .models import Document

class DocumentUploadForm(forms.ModelForm):
    """
    Form for uploading a new document for chunking and indexing.
    """
    class Meta:
        model = Document
        fields = ['file']

    def save(self, commit=True):
        # Override save to set document name
        instance = super().save(commit=False)
        uploaded_file = self.cleaned_data.get('file')
        if uploaded_file:
            instance.name = uploaded_file.name
        if commit:
            instance.save()
        return instance
