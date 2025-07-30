import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("WEATHER_API_KEY")
BASE_URL = "http://api.weatherapi.com/v1/current.json"

def get_weather(city):
    params = {
        "key": API_KEY,
        "q": city,
        "aqi": "no"
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        location = data["location"]["name"]
        country = data["location"]["country"]
        temp = data["current"]["temp_f"]
        condition = data["current"]["condition"]["text"]

        print(f"\n Location: {location}, {country}")
        print(f" Temperature: {temp}°F")
        print(f" Condition: {condition}")

    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e)
    except Exception as e:
        print("Something went wrong:", e)


mode = "dev"

if mode == "prod":
    get_weather("Bolivia")  # hardcoded test
else:
    city = input("Enter a city: ")
    get_weather(city)
