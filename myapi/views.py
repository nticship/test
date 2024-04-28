import random
from django.shortcuts import render

# Create your views here.

import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from rest_framework import status

from PIL import Image
import io

import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class RootView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request, *args, **kwargs):
        return Response({
            'message': 'Root',
        })

class ImageUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        image = request.FILES['image']

        # Save the uploaded image to a temporary location (or any other location you prefer)
        filename = default_storage.save('images/' + image.name, ContentFile(image.read()))

        # You can process the image here if needed, then respond with JSON
        return Response({
            'message': 'Image uploaded successfully',
            'filename': filename,
            'size': image.size,
            'content_type': image.content_type,
        })
    





# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


class Model_Loader:
    
    def __init__(self):
        self.model = self.build_model()
    
    def build_model(self):

        model = tf.keras.Sequential()
        model.add(layers.Resizing(150, 150))
        model.add(layers.Rescaling(1/255.0))

        model.add(layers.Flatten())
        model.add(layers.Dense(units=256, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())

        model.add(layers.Dense(units=11, activation='softmax'))
        
        return model


class ImageClassification(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)

        # Get the uploaded image file
        image = request.FILES['image']

        filename = default_storage.save('myapi/images/' + image.name, ContentFile(image.read()))





        # img_path = os.path.join('./dataset/snow/0830.jpg')
        img_path = filename


        

        df_single_image = pd.DataFrame({
            'filepaths': [img_path],
            'labels': [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(5, 9))) for _ in range(1)]
        })

        # Data generator with minimal preprocessing for a single image
        test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True,
            zoom_range=0.2,
            rotation_range=20,
            shear_range=0.2
        )

        # Create a data generator for the single image
        image_pret = test_datagen.flow_from_dataframe(
            df_single_image,
            x_col='filepaths',
            y_col=None,  # No class information, because it's unknown
            target_size=(224, 224),
            batch_size=1,
            class_mode=None  # No specific class mode
        )

        





        # Prediction logic
        model = {
            'vgg16': Model_Loader().build_model(),
            'resnet50': Model_Loader().build_model(),
            'xception': Model_Loader().build_model(),
        }

        model['vgg16'].load_weights('myapi/ai_models/vgg16_weights.h5', by_name=True)
        model['resnet50'].load_weights('myapi/ai_models/resnet50_weights.h5', by_name=True)
        model['xception'].load_weights('myapi/ai_models/xception_weights.h5', by_name=True)

        model['vgg16'].compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        model['resnet50'].compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        model['xception'].compile(
            optimizer='adam', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )








        # Get predictions
        prediction = {
            'vgg16': model['vgg16'].predict(image_pret)[0],
            'resnet50': model['resnet50'].predict(image_pret)[0],
            'xception': model['xception'].predict(image_pret)[0],
        }

        predicted_class_index = {
            'vgg16': np.argmax(prediction['vgg16']),
            'resnet50': np.argmax(prediction['resnet50']),
            'xception': np.argmax(prediction['xception']),
        }

        classes = {
            0: 'dew',
            1: 'fogsmog',
            2: 'frost',
            3: 'glaze',
            4: 'hail',
            5: 'lightning',
            6: 'rain',
            7: 'rainbow',
            8: 'rime',
            9: 'sandstorm',
            10: 'snow'
        }

        predicted_class = {
            'vgg16': classes[predicted_class_index['vgg16']],
            'resnet50': classes[predicted_class_index['resnet50']],
            'xception': classes[predicted_class_index['xception']],
        }









        return Response({
            'message': 'Image uploaded successfully',

            'filename': image.name,
            'size': image.size,
            'content_type': image.content_type,

            'vgg16': predicted_class['vgg16'],
            'resnet50': predicted_class['resnet50'],
            'xception': predicted_class['xception'],

            'predictions_de_vgg16': str(prediction['vgg16']),
            'predictions_de_resnet50': str(prediction['resnet50']),
            'predictions_de_xception': str(prediction['xception']),
        })

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------










































# from io import BytesIO
# from django.core.files.uploadedfile import InMemoryUploadedFile

# import numpy as np
# from keras.models import load_model, load_weights
# from keras.preprocessing.image import load_img, img_to_array

# import tensorflow as tf
# from keras import Sequential, layers, callbacks


# from keras.utils import custom_object_scope

# # Function to load and preprocess the image
# def preprocess_image(image_file):
#     # Convert InMemoryUploadedFile to BytesIO for compatibility with load_img
#     if isinstance(image_file, InMemoryUploadedFile):
#         image_file = BytesIO(image_file.read())

#     # Load the image with the appropriate target size
#     img = load_img(image_file, target_size=(150, 150))  # Adjust target_size to your model's needs
#     img_array = img_to_array(img)  # Convert the image to an array
#     img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit model input
#     return img_array

# EPOCHS = 10
# height, width = 150, 150

# class Model_builder:
    
#     def __init__(self, pretrained_model):
#         self.model = self.build_model(pretrained_model)
    
#     def build_model(self, pretrained_model):

#         for layer in pretrained_model.layers:
#             layer.trainable = False

#         model = tf.keras.Sequential()
#         model.add(layers.Resizing(height, width))
#         model.add(layers.Rescaling(1/255.0))
#         model.add(pretrained_model)

#         model.add(layers.Flatten())
#         model.add(layers.Dense(units=256, activation='relu'))
#         model.add(layers.Dropout(0.2))
#         model.add(layers.BatchNormalization())

#         model.add(layers.Dense(units=11, activation='softmax'))
        
#         return model
    
#     def compile_model(self, opt):
#         self.model.compile(
#             optimizer = opt,
#             loss = 'categorical_crossentropy',
#             metrics = ['accuracy']
#         )
    
#     def model_summary(self):
#         self.model.build((None, height, width, 3))
#         self.model.summary()
        
#     def get_model(self):
#         return self.model
    
    
# # using tensorflow i want to load a models that i devoloped and choose whether the image that i'm getting throw the url is what
# class ImageClassification(APIView):
#     parser_classes = [MultiPartParser, FormParser]

#     def post(self, request, *args, **kwargs):
#      #    image_url = 'http://localhost:8000/myapi/ai_models/2208.jpg'
#      #    return Response({
#      #      'message': 'Image uploaded and processed successfully',
#      #      'image_url': image_url
#      #    })
    
#         if 'image' not in request.FILES:
#             return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
#         image = request.FILES['image']

#         # Load and preprocess the image
#         img_array = preprocess_image(image)

        
#         # Load the model(s)
#         model = {
#           'vgg16': load_weights('myapi/ai_models/vgg16_model.h5'),
#           # 'resnet50': load_model('myapi/ai_models/resnet_model.h5'),
#           # 'xception': load_model('myapi/ai_models/xception_model.h5'),
#         }

#         vgg_pretrained = tf.keras.applications.VGG16(weights=model['vgg16'], include_top=False, input_shape=(height, width, 3))
#         vgg_model = Model_builder(vgg_pretrained)
#         vgg_model.compile_model('adam')
#         vgg_model.model_summary()
#         vgg_model = vgg_model.get_model()

#      #    model=load_model('myapi/ai_models/xception_model.h5')

#         # Get prediction from the desired model (change 'xception' to the appropriate key)
#         prediction = vgg_model.predict(img_array)[0][0]

#         # Save the uploaded image to a temporary location (if needed)
#         filename = default_storage.save('images/' + image.name, ContentFile(image.read()))

#         # Respond with results
#         return Response({
#             'message': 'Image uploaded and processed successfully',
#             'filename': filename,
#             'size': image.size,
#             'content_type': image.content_type,
#             'prediction': str(prediction)  # You can round or format this prediction
#         })
#      #    prediction = {
#      #      # 'vgg16':model['vgg16'].predict(img_array)[0][0],
#      #      # 'resnet50':model['resnet50'].predict(img_array)[0][0],
#      #      'xception':model['xception'].predict(img_array)[0][0]
#      #    }

#      #    # Save the uploaded image to a temporary location (or any other location you prefer)
#      #    filename = default_storage.save('images/' + image.name, ContentFile(image.read()))

#      #    # You can process the image here if needed, then respond with JSON
#      #    return Response({
#      #        'message': 'Image uploaded successfully and processed successfully',
#      #        'filename': filename,
#      #        'size': image.size,
#      #        'content_type': image.content_type,
#      #      #   'prediction de vgg16': str(prediction['vgg16']),
#      #      #   'prediction de resnet50': str(prediction['resnet50']),
#      #        'prediction de xception': str(prediction['xception'])
#      #    })