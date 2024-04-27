from django.urls import path
from . import views

urlpatterns = [
    path('', views.RootView.as_view(), name='root-view'),
    path('test', views.ImageUploadView.as_view(), name='image-upload'),  # URL for image upload
    path('classify', views.ImageClassification.as_view(), name='image-classify'),  # URL for image upload
]