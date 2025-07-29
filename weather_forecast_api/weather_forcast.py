import requests
api_key = "your_api_key_here"
base_url = "http://api.weatherapi.com/v1/current.json"
city = "Las Vegas"
url = f"{base_url}?key={api_key}&q={city}"
