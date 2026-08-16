import json

import requests

# I like to set the url as a variable so it's easier to use in subsequent code
url = "http://127.0.0.1:8000"

# get request
r = requests.get(url)

# print status code
print(f"Status Code: {r.status_code}")
# print the welcome message
print(f"Result: {r.json()['message']}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# send a POST using the data above
r = requests.post(f"{url}/data/", json=data)

# print the status code
print(f"Status Code: {r.status_code}")
# print the result
print(f"Result: {r.json()['result']}")
