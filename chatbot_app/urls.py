from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register_view, name='register_view'),
    path('login/', views.login_view, name='login_view'),
    path('logout/', views.logout_view, name='logout_view'),
    path('chat/', views.chatbot_view, name='chatbot_view'),
    path('chat_api/', views.chat_api, name='chat_api'),
    path('chat_api/stream/', views.chat_api_stream, name='chat_api_stream'),
    path('profile_edit/', views.profile_edit, name='profile_edit'),
    path('password_change/', views.password_change, name='password_change'),
    path('find_username/', views.find_username, name='find_username'),
    path('find_password/', views.find_password, name='find_password'),
    path('', views.home, name='home'),
]
