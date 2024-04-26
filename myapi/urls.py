from django.urls import path
from . import views

urlpatterns = [
    path('test', views.ImageUploadView.as_view(), name='image-upload'),  # URL for image upload
]