"""
URL configuration for kiepnandaluc22h02 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from home import views
from django.conf import settings
from django.conf.urls.static import static




urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.get_home, name='home'),
    path('read_image/', views.read_image, name='read_image'),
    path('test1/', views.test_page, name='test_page'),
    path('test2/', views.test_page2, name='test_page2'),
    path('test3/', views.test_page3, name='test_page3'),
    path('test4/', views.test_page4, name='test_page4'),
    path('test5/', views.test_page5, name='test_page5'),
    path('bg_removal/', views.bg_removal, name='bg_removal'),
    path('simple_threshold/', views.simple_threshold, name='simple_threshold'),
    path('otsu_threshold/', views.otsu_threshold, name='otsu_threshold'),
    path('noise_removal/', views.noise_removal, name='noise_removal'),
    path('preprocess_otsuTH_image/', views.preprocess_otsuTH_image, name='preprocess_otsuTH_image'),
    path('preprocess_STH_image/', views.preprocess_STH_image, name='preprocess_STH_image'),
    path('render_latex_to_png/', views.render_latex_to_png, name='render_latex_to_png'),
    path('read_symbol_from_image/', views.read_symbol_from_image, name='read_symbol_from_image'),
    path('DefineTags/', views.DefineTags, name='DefineTags'),
    path('preprocess_latex/', views.preprocess_latex, name='preprocess_latex'),
] +  static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)