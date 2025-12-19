from insightface.app import FaceAnalysis
import os
import numpy as np
from PIL import Image

# Load model
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

def extract_face_embed(image_path):
    image = Image.open(image_path).convert("RGB")
    img = np.array(image)[:, :, ::-1]  # RGB -> BGR

    # Run face analysis
    faces = app.get(img)

    print(f"Detected faces: {len(faces)}")
    face = faces[0]
    face = max(faces, key=lambda x: x.det_score)
    bbox = face.bbox.astype(int)
    face_img  = image.crop(bbox)
    embedding = face.embedding
    img_name = os.path.basename(image_path)
    face_img.save(f"extracted_{img_name}")
    return embedding

if __name__ == "__main__":
    embedding = extract_face_embed("dataset/girl/01.jpg")
    print("Face embedding shape:", embedding.shape)
    
