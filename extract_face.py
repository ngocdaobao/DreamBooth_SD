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
    img = Image.open(image_path).convert("RGB")
    img = np.array(img)[:, :, ::-1]  # RGB -> BGR

    # Run face analysis
    faces = app.get(img)

    print(f"Detected faces: {len(faces)}")
    #Crop face and save embedding
    if len(faces) > 0:
        raise NotImplementedError("Multiple faces detected. This function only handles single face extraction.")

    face = faces[0]
    #Save face crop
    bbox = map(int, face.bbox)
    face_crop = img.crop(bbox)
    face_crop.save("extracted_face.jpg")
    embedding = face.embedding  # shape: (512,)
    return embedding

if __name__ == "__main__":
    embedding = extract_face_embed("dataset\girl\01.jpg")
    print("Face embedding shape:", embedding.shape)
    
