import googlemaps
from secret import key

client = googlemaps.Client(key)
places = googlemaps.places.places_nearby(client, (40.63, 22.94), 500)
for place in places['results']:
    loc = place['geometry']['location']
    lat, lon = loc['lat'], loc['lng']
    name = place['name']
    types = place['types']
    rating = place.get('rating')
    photos = place.get('photos')
    print(name, types)
