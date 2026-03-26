from django.shortcuts import render
from decimal import Decimal, InvalidOperation
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.http import require_GET, require_POST
from django.views.decorators.csrf import csrf_exempt
from .streams import generate_stream_detections, generate_multi_camera_stream
from django.db import transaction
from .models import Camera, Incident, Camera_Incident
import math
import os

ACCIDENT_MODEL_URL = os.getenv("ACCIDENT_MODEL_URL")
FIRE_MODEL_URL     = os.getenv("FIRE_MODEL_URL")
TIME_QUANTUM       = int(os.getenv("TIME_QUANTUM", 3))


@require_GET
def stream_detect(request):
    youtube_url  = request.GET.get("url", "").strip()
    live         = request.GET.get("live", "true").lower() == "true"
    time_quantum = int(request.GET.get("tq", TIME_QUANTUM))

    if not youtube_url:
        return JsonResponse({"error": "url parameter is required."}, status=400)

    gen = generate_stream_detections(
        youtube_url        = youtube_url,
        live               = live,
        time_quantum       = time_quantum,
        accident_model_url = ACCIDENT_MODEL_URL,
        fire_model_url     = FIRE_MODEL_URL
    )

    return StreamingHttpResponse(gen, content_type="application/x-ndjson")


@require_GET
def stream_all_cameras(request):
    time_quantum = int(request.GET.get("tq", TIME_QUANTUM))
    live         = request.GET.get("live", "true").lower() == "true"

    cameras = [
        {
            "id":        cam.id,
            "latitude":  str(cam.latitude),
            "longitude": str(cam.longitude),
            "url":       cam.live_feed_url
        }
        for cam in Camera.objects.filter(live=live)
    ]

    gen = generate_multi_camera_stream(
        cameras            = cameras,
        accident_model_url = ACCIDENT_MODEL_URL,
        fire_model_url     = FIRE_MODEL_URL,
        time_quantum       = time_quantum,
        live               = live
    )

    return StreamingHttpResponse(gen, content_type="application/x-ndjson")


@csrf_exempt
@require_POST
def create_camera(request):
    try:
        latitude      = request.POST.get("latitude",      "").strip()
        longitude     = request.POST.get("longitude",     "").strip()
        live_feed_url = request.POST.get("live_feed_url", "").strip()
        live_raw      = request.POST.get("live",          "").strip().lower()

        if not latitude or not longitude or not live_feed_url:
            return JsonResponse({
                "success": False,
                "error":   "latitude, longitude and live_feed_url are all required."
            }, status=400)

        try:
            latitude  = float(latitude)
            longitude = float(longitude)
        except ValueError:
            return JsonResponse({
                "success": False,
                "error":   "latitude and longitude must be valid numbers."
            }, status=400)

        if not (-90 <= latitude <= 90):
            return JsonResponse({
                "success": False,
                "error":   "latitude must be between -90 and 90."
            }, status=400)

        if not (-180 <= longitude <= 180):
            return JsonResponse({
                "success": False,
                "error":   "longitude must be between -180 and 180."
            }, status=400)

        if live_raw in ("", "true", "1"):
            live = True
        elif live_raw in ("false", "0"):
            live = False
        else:
            return JsonResponse({
                "success": False,
                "error":   "live must be true or false."
            }, status=400)

        if Camera.objects.filter(live_feed_url=live_feed_url).exists():
            return JsonResponse({
                "success": False,
                "error":   "a camera with this live feed URL already exists."
            }, status=409)

        if Camera.objects.filter(latitude=latitude, longitude=longitude).exists():
            return JsonResponse({
                "success": False,
                "error":   "a camera already exists at these exact coordinates."
            }, status=409)

        camera = Camera.objects.create(
            latitude      = latitude,
            longitude     = longitude,
            live_feed_url = live_feed_url,
            live          = live
        )

        return JsonResponse({
            "success": True,
            "message": "Camera created successfully.",
            "camera": {
                "id":            camera.id,
                "latitude":      str(camera.latitude),
                "longitude":     str(camera.longitude),
                "live_feed_url": camera.live_feed_url,
                "live":          camera.live
            }
        }, status=201)

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error":   f"An unexpected error occurred: {str(e)}"
        }, status=500)

def test(request):
    return render(request, "main/temp.html")


def dashboard(request):
    return render(request, "main/dashboard.html")


@require_GET
def delete_all_cameras(request):
    count, _ = Camera.objects.all().delete()
    return JsonResponse({
        "success": True,
        "message": f"Deleted {count} camera(s) from the database."
    })


@csrf_exempt
@require_POST
def create_incident(request):
    try:
        latitude      = request.POST.get("latitude",      "").strip()
        longitude     = request.POST.get("longitude",     "").strip()
        incident_type = request.POST.get("incident_type", "").strip()
        description   = request.POST.get("description",   "").strip()
        date_created  = request.POST.get("date_created",  "").strip()

        if not latitude or not longitude or not incident_type or not date_created:
            return JsonResponse({
                "success": False,
                "error":   "latitude, longitude, incident_type and date_created are all required."
            }, status=400)

        try:
            latitude  = float(latitude)
            longitude = float(longitude)
        except ValueError:
            return JsonResponse({
                "success": False,
                "error":   "latitude and longitude must be valid numbers."
            }, status=400)

        if not (-90 <= latitude <= 90):
            return JsonResponse({
                "success": False,
                "error":   "latitude must be between -90 and 90."
            }, status=400)

        if not (-180 <= longitude <= 180):
            return JsonResponse({
                "success": False,
                "error":   "longitude must be between -180 and 180."
            }, status=400)

        incident = Incident.objects.create(
            latitude      = latitude,
            longitude     = longitude,
            incident_type = incident_type,
            description   = description or None,
            date_created  = date_created
        )

        return JsonResponse({
            "success": True,
            "message": "Incident created successfully.",
            "incident": {
                "id":            incident.id,
                "latitude":      str(incident.latitude),
                "longitude":     str(incident.longitude),
                "incident_type": incident.incident_type,
                "description":   incident.description,
                "date_created":  str(incident.date_created)
            }
        }, status=201)

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error":   f"An unexpected error occurred: {str(e)}"
        }, status=500)


@require_GET
def delete_all_incidents(request):
    count, _ = Incident.objects.all().delete()
    return JsonResponse({
        "success": True,
        "message": f"Deleted {count} incident(s) from the database."
    })


def get_incidents_within_radius(latitude: float, longitude: float, distance_km: float):
    lat_delta = distance_km / 111.32
    lon_delta = distance_km / (111.32 * math.cos(math.radians(latitude)))

    min_lat = latitude  - lat_delta
    max_lat = latitude  + lat_delta
    min_lon = longitude - lon_delta
    max_lon = longitude + lon_delta

    candidates = Incident.objects.filter(
        latitude__gte  = min_lat,
        latitude__lte  = max_lat,
        longitude__gte = min_lon,
        longitude__lte = max_lon,
    )

    EARTH_RADIUS_KM = 6371.0

    def haversine(lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
        return EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))

    return [
        incident for incident in candidates
        if haversine(
            latitude, longitude,
            float(incident.latitude), float(incident.longitude)
        ) <= distance_km
    ]


@csrf_exempt
@require_GET
def incidents_within_radius(request):
    try:
        latitude    = request.GET.get("latitude",    "").strip()
        longitude   = request.GET.get("longitude",   "").strip()
        distance_km = request.GET.get("distance_km", "").strip()

        if not latitude or not longitude or not distance_km:
            return JsonResponse({
                "success": False,
                "error":   "latitude, longitude and distance_km are all required."
            }, status=400)

        try:
            latitude    = float(latitude)
            longitude   = float(longitude)
            distance_km = float(distance_km)
        except ValueError:
            return JsonResponse({
                "success": False,
                "error":   "latitude, longitude and distance_km must all be valid numbers."
            }, status=400)

        if not (-90 <= latitude <= 90):
            return JsonResponse({
                "success": False,
                "error":   "latitude must be between -90 and 90."
            }, status=400)

        if not (-180 <= longitude <= 180):
            return JsonResponse({
                "success": False,
                "error":   "longitude must be between -180 and 180."
            }, status=400)

        if distance_km <= 0:
            return JsonResponse({
                "success": False,
                "error":   "distance_km must be a positive number."
            }, status=400)

        incidents = get_incidents_within_radius(latitude, longitude, distance_km)

        return JsonResponse({
            "success":        True,
            "searched_at":    {"latitude": latitude, "longitude": longitude},
            "distance_km":    distance_km,
            "incident_count": len(incidents),
            "incidents": [
                {
                    "id":            incident.id,
                    "latitude":      str(incident.latitude),
                    "longitude":     str(incident.longitude),
                    "incident_type": incident.incident_type,
                    "description":   incident.description,
                    "date_created":  str(incident.date_created),
                }
                for incident in incidents
            ]
        }, status=200)

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error":   f"An unexpected error occurred: {str(e)}"
        }, status=500)


@csrf_exempt
@require_POST
def delete_by_coordinates(request):
    latitude_raw  = request.POST.get("latitude",  "").strip()
    longitude_raw = request.POST.get("longitude", "").strip()
    to_delete     = request.POST.get("to_delete", "").strip().lower()

    if not latitude_raw or not longitude_raw or not to_delete:
        return JsonResponse({
            "success": False,
            "error":   "latitude, longitude and to_delete are all required."
        }, status=400)

    try:
        latitude  = Decimal(latitude_raw)
        longitude = Decimal(longitude_raw)
    except (InvalidOperation, TypeError):
        return JsonResponse({
            "success": False,
            "error":   "latitude and longitude must be valid decimal numbers."
        }, status=400)

    if latitude < Decimal("-90") or latitude > Decimal("90"):
        return JsonResponse({
            "success": False,
            "error":   "latitude must be between -90 and 90."
        }, status=400)

    if longitude < Decimal("-180") or longitude > Decimal("180"):
        return JsonResponse({
            "success": False,
            "error":   "longitude must be between -180 and 180."
        }, status=400)

    if max(0, -latitude.as_tuple().exponent) > 6 or max(0, -longitude.as_tuple().exponent) > 6:
        return JsonResponse({
            "success": False,
            "error":   "latitude and longitude must not have more than 6 decimal places."
        }, status=400)

    try:
        with transaction.atomic():
            if to_delete == "camera":
                obj        = Camera.objects.filter(latitude=latitude, longitude=longitude).first()
                model_name = "Camera"
            elif to_delete == "incident":
                obj        = Incident.objects.filter(latitude=latitude, longitude=longitude).first()
                model_name = "Incident"
            elif to_delete in {"camera_incident", "camera-incident", "camera incident"}:
                obj        = Camera_Incident.objects.filter(
                                 camera__latitude=latitude,
                                 camera__longitude=longitude
                             ).first()
                model_name = "Camera_Incident"
            else:
                return JsonResponse({
                    "success": False,
                    "error":   "to_delete must be one of: camera, incident, camera_incident."
                }, status=400)

            if not obj:
                return JsonResponse({
                    "success": False,
                    "message": f"No matching {model_name} record found for the given coordinates."
                }, status=404)

            deleted_id = obj.id
            obj.delete()

        return JsonResponse({
            "success": True,
            "message": f"{model_name} record deleted successfully.",
            "deleted": {
                "id":        deleted_id,
                "model":     model_name,
                "latitude":  str(latitude),
                "longitude": str(longitude)
            }
        }, status=200)

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error":   f"An unexpected error occurred: {str(e)}"
        }, status=500)
    
@csrf_exempt
@require_POST
def delete_all_data(request): # DELETE DURING PRODUCTION
    try:
        with transaction.atomic():
            camera_incident_count, _ = Camera_Incident.objects.all().delete()
            incident_count, _        = Incident.objects.all().delete()
            camera_count, _          = Camera.objects.all().delete()

        return JsonResponse({
            "success": True,
            "message": "All data deleted successfully.",
            "deleted_counts": {
                "camera_incidents": camera_incident_count,
                "incidents": incident_count,
                "cameras": camera_count
            }
        }, status=200)

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }, status=500)
    
@csrf_exempt
@require_POST
def delete_all_camera_incidents(request):
    try:
        with transaction.atomic():
            count, _ = Camera_Incident.objects.all().delete()

        return JsonResponse({
            "success": True,
            "message": f"Deleted {count} camera incident(s).",
            "deleted_count": count
        }, status=200)

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}"
        }, status=500)

@require_GET
def get_all_cameras(request):
    cameras = Camera.objects.all()
    data = [
        {
            "id": cam.id,
            "latitude": str(cam.latitude),
            "longitude": str(cam.longitude),
            "live_feed_url": cam.live_feed_url,
            "live": cam.live
        }
        for cam in cameras
    ]
    return JsonResponse({"success": True, "cameras": data}, status=200)


@require_GET
def get_all_incidents(request):
    incidents = Incident.objects.all()
    data = [
        {
            "id": incident.id,
            "latitude": str(incident.latitude),
            "longitude": str(incident.longitude),
            "incident_type": incident.incident_type,
            "description": incident.description,
            "date_created": incident.date_created.isoformat() if incident.date_created else None
        }
        for incident in incidents
    ]
    return JsonResponse({"success": True, "incidents": data}, status=200)


@require_GET
def get_all_camera_incidents(request):
    # select_related optimizes the database query to fetch camera data simultaneously
    camera_incidents = Camera_Incident.objects.select_related('camera').all()
    data = [
        {
            "id": ci.id,
            "incident_type": ci.incident_type,
            "date_created": ci.date_created.isoformat() if ci.date_created else None,
            "footage": ci.footage,
            "camera_details": {
                "id": ci.camera.id,
                "latitude": str(ci.camera.latitude),
                "longitude": str(ci.camera.longitude),
                "live_feed_url": ci.camera.live_feed_url,
                "live": ci.camera.live
            }
        }
        for ci in camera_incidents
    ]
    return JsonResponse({"success": True, "camera_incidents": data}, status=200)


@require_GET
def get_one_by_coordinates(request):
    latitude_raw  = request.GET.get("latitude", "").strip()
    longitude_raw = request.GET.get("longitude", "").strip()

    if not latitude_raw or not longitude_raw:
        return JsonResponse({
            "success": False,
            "error": "latitude and longitude are required."
        }, status=400)

    try:
        latitude = Decimal(latitude_raw)
        longitude = Decimal(longitude_raw)
    except (InvalidOperation, TypeError):
        return JsonResponse({
            "success": False,
            "error": "latitude and longitude must be valid decimal numbers."
        }, status=400)

    camera = Camera.objects.filter(latitude=latitude, longitude=longitude).first()
    
    if not camera:
        return JsonResponse({
            "success": False,
            "message": "No camera found at these coordinates.",
            "camera": None
        }, status=404)

    # Fetch all incidents linked to this specific camera, sorted by newest first
    incidents = Camera_Incident.objects.filter(camera=camera).order_by('-date_created')

    camera_data = {
        "id": camera.id,
        "latitude": str(camera.latitude),
        "longitude": str(camera.longitude),
        "live_feed_url": camera.live_feed_url,
        "live": camera.live
    }

    incident_data = [
        {
            "id": ci.id,
            "incident_type": ci.incident_type,
            "date_created": ci.date_created.isoformat() if ci.date_created else None,
            "footage": ci.footage
        }
        for ci in incidents
    ]

    return JsonResponse({
        "success": True,
        "camera": camera_data,
        "camera_incidents": incident_data
    }, status=200)