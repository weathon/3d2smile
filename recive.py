#GPT Used
from fastapi import FastAPI
import base64
import io
from PIL import Image
import os 
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import time
from pydantic import BaseModel

class ImagePOST(BaseModel):
    name: str
    base64: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set the appropriate origin or origins instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html","r") as f:
        return f.read()

@app.post("/save_image")
async def save_image(img: ImagePOST):
    # Decode base64 image
    base64_image = img.base64.split(",")[1]
    filename = img.name
    image_data = base64.b64decode(base64_image)

    # Open image using PIL
    file = io.BytesIO(image_data)
    image = Image.open(file)
    
    # Save image as PNG
    # image.save(f"../pngs/{filename}.png", "PNG")
    
    return {"message": "Image saved successfully."}

@app.get("/all_images")
def images():
    return os.listdir("SDFs")[:100]


@app.get("/get_SDF")
def sdf(filename):
    time.sleep(0.2)
    with open("SDFs/"+filename,"r") as f:
        return f.read() 