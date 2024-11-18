from django.urls import path,include
from .views import searchQuery
urlpatterns = [
    path('', searchQuery, name='search'), 
]
