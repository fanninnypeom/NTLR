import pickle
import json
rawData = pickle.load(open('/data/wuning/NTLR/beijing/beijingTaxi', 'rb'))
traJson = {}
traJson["type"] = "FeatureCollection"
traJson["name"] = "trips"
traJson["crs"] = {}
traJson["crs"]["type"] = "name"
traJson["crs"]["properties"] = {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
traJson["features"] = []
count = 1
for tra in rawData:
  temp = { "type": "Feature", "properties": { "id": count }, "geometry": { "type": "LineString", "coordinates": [] } }  
  temp["geometry"]["coordinates"] = tra
  traJson["features"].append(temp)
  count += 1
with open("/data/wuning/NTLR/beijing/beijing_tras.json","w") as f:
  json.dump(traJson, f)

