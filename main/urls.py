from django.urls import path
from . import views

urlpatterns = [
    path("stream/detect/",      views.stream_detect,      name="stream_detect"),
    path("stream/cameras/",     views.stream_all_cameras, name="stream_all_cameras"),
    path("camera/create/",      views.create_camera,      name="create_camera"),
    path("test/",               views.test,               name="test"),
    path("dashboard/",          views.dashboard,          name="dashboard"),
    path("camera/delete-all/", views.delete_all_cameras, name="delete_all_cameras"),
    path("incident/delete-all/", views.delete_all_incidents, name="delete_all_incidents"),
    path("incident/create/",      views.create_incident,      name="create_incident"),
    path("incident/within-radius/", views.incidents_within_radius, name="incidents_within_radius"),
    path("delete-by-coordinates/", views.delete_by_coordinates, name="delete_by_coordinates"),
    path('clean-database/', views.delete_all_data, name='clean_database'),
    path('delete-all-camera-incidents/', views.delete_all_camera_incidents, name='delete_all_camera_incidents'),
    

]