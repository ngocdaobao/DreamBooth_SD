from insightface.app import FaceAnalysis

import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image

# Load model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_face_embed(image_path: str):
    image = Image.open(image_path).convert("RGB")
    img = np.array(image)[:, :, ::-1]  # RGB -> BGR

    # Run face analysis
    faces = app.get(img)

    print(f"Detected faces: {len(faces)}")
    #Crop face and save embedding
    if len(faces) > 0:
        #Select face with highest detection score
        face = max(faces, key=lambda x: x.det_score)
        bbox = face.bbox.astype(int)
        face_img  = image.crop(bbox)
        embedding = face.embedding

    return embedding

if __name__ == "__main__":
    embedding = extract_face_embed("dataset/girl/01.jpg")
    print("Face embedding shape:", embedding.shape)
    
