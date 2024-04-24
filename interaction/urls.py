from django.urls import path
from interaction import views

urlpatterns = [
    path('', views.index, name='index'),

    # Other URL patterns
]
