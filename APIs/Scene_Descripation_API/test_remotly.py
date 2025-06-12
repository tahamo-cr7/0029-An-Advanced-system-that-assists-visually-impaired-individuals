import requests
import cv2

# Define the server URL and image path
url = "https://395c-197-38-78-133.ngrok-free.app/caption/"
image_path = "1.jpg"

# Read the image using OpenCV
image = cv2.imread(image_path)

# Send the image to the server
with open(image_path, "rb") as img:
    files = {"file": img}
    response = requests.post(url, files=files)

# Extract the caption
caption = response.json().get("caption", "No caption received")
print("Caption:", caption)

# Add caption text on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, caption, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

# Show the image in a window
cv2.imshow("Image with Caption", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
