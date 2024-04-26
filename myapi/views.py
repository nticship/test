from django.shortcuts import render

# Create your views here.

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

class ImageUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        image = request.FILES['image']

        # Save the uploaded image to a temporary location (or any other location you prefer)
        filename = default_storage.save(image.name, ContentFile(image.read()))

        # You can process the image here if needed, then respond with JSON
        return Response({
            'message': 'Image uploaded successfully',
            'filename': filename,
            'size': image.size,
            'content_type': image.content_type,
        })
