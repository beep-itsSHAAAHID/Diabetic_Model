import requests

# Define the URL of the Flask server
url = 'http://127.0.0.1:5000/predict'

# Load an image file to send to the server
image_path = "C:\\Users\\madat\\OneDrive\\Desktop\\Retinal Dataset\\Severe DR\\Severe DR_167.png"
files = {'image': open(image_path, 'rb')}

# Send a POST request to the server
response = requests.post(url, files=files)

# Print the response from the server
print(response.json())
    