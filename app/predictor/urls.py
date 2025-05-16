from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/individual/', views.predict_individual, name='predict_individual'),
    path('predict/batch/', views.predict_batch, name='predict_batch'),
]
