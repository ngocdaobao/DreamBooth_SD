import clip
import torch
from PIL import Image
import torch.nn.functional as F

def im2im(img_real, img_gen, model, preprocess, device='cuda'):
    # Load images
    img_real = preprocess(img_real).unsqueeze(0).to(device)
    img_gen = preprocess(img_gen).unsqueeze(0).to(device)

    with torch.no_grad():
        emb_real = model.encode_image(img_real)
        emb_gen = model.encode_image(img_gen)

    # Normalize (cosine similarity)
    emb_real = emb_real / emb_real.norm(dim=-1, keepdim=True)
    emb_gen = emb_gen / emb_gen.norm(dim=-1, keepdim=True)

    similarity = (emb_real @ emb_gen.T).item()  
    return similarity

def im2prompt(prompt, img, model, preprocess, device='cuda'):
    # Preprocess image
    img = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        text_tokens = clip.tokenize([prompt]).to(device)
        emb_text = model.encode_text(text_tokens)
        emb_img = model.encode_image(img)

    # Normalize (cosine similarity)
    emb_text = emb_text / emb_text.norm(dim=-1, keepdim=True)
    emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)

    similarity = (emb_text @ emb_img.T).item()  
    return similarity

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Example usage
    img_real = Image.open("real_image.png")
    img_gen = Image.open("generated_image.png")
    prompt = "A description of the image"

    sim_im2im = im2im(img_real, img_gen, model, preprocess, device)
    sim_im2prompt = im2prompt(prompt, img_gen, model, preprocess, device)

    print(f"Image-to-Image Similarity: {sim_im2im}")
    print(f"Image-to-Prompt Similarity: {sim_im2prompt}")
