# Taxis

The goal is to predict the route price based on the starting longitude, latitude and time.

## Features

We are given:

* starting latitude
* starting longitude
* starting time

(I tested weekday and month. Those time variables did not seem very relevant.)

Potential features:

* distance to several random points
* discretization of lon/lat in numbers (and then hot-encoding for sklearn)
* places around each geographical point

We might be able to get information on the various locations using the API: https://developers.google.com/places/web-service/details

## Links

Competition: https://web.fe.up.pt/~epia2017/ai-competitions/discovery-challenge/
