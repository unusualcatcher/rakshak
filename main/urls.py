from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path("stream/detect/", views.stream_detect, name="stream_detect"),
    path("stream/cameras/", views.stream_all_cameras, name="stream_all_cameras"),
    
    # --- Create Endpoints ---
    path("camera/create/", views.create_camera, name="create_camera"),
    path("incident/create/", views.create_incident, name="create_incident"),
    
    # --- Get All Endpoints ---
    path("camera/get-all/", views.get_all_cameras, name="get_all_cameras"),
    path("incident/get-all/", views.get_all_incidents, name="get_all_incidents"),
    path("camera-incident/get-all/", views.get_all_camera_incidents, name="get_all_camera_incidents"),
    
    # --- Get One Endpoints ---
    path("camera/get-one/", views.get_one_by_coordinates, name="get_one_by_coordinates"),
    path("incident/within-radius/", views.incidents_within_radius, name="incidents_within_radius"),
    
    # --- Delete Endpoints ---
    path("camera/delete-all/", views.delete_all_cameras, name="delete_all_cameras"),
    path("incident/delete-all/", views.delete_all_incidents, name="delete_all_incidents"),
    path("delete-all-camera-incidents/", views.delete_all_camera_incidents, name="delete_all_camera_incidents"),
    path("delete-by-coordinates/", views.delete_by_coordinates, name="delete_by_coordinates"),
    path("clean-database/", views.delete_all_data, name="clean_database"),
    
    # --- Miscellaneous Endpoints ---
    path("test/", views.test, name="test"),
    path("dashboard/", views.dashboard, name="dashboard"),
]

urlpatterns += static('/footages/', document_root=settings.BASE_DIR / 'footages')