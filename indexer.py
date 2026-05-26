import os
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb
from tqdm import tqdm

# ==========================================
# 1. PATH & ENGINE CONFIGURATION
# ==========================================
CSV_PATH = r"C:\Users\Lalisha\Desktop\NEW_Project_multiModel\fashion-dataset\styles.csv"
IMAGES_DIR = r"C:\Users\Lalisha\Desktop\NEW_Project_multiModel\fashion-dataset\images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection(name="fashion_retrieval")
    print("🧹 Old database collection wiped clean.")
except Exception:
    pass

collection = chroma_client.get_or_create_collection(
    name="fashion_retrieval", 
    metadata={"hnsw:space": "cosine"}
)

# ==========================================
# 2. METADATA LOADING
# ==========================================
print("Loading styles.csv metadata...")
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
df = df.dropna(subset=['id', 'productDisplayName'])
df['id'] = df['id'].astype(int)
df['image_path'] = df['id'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))

df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Total matching images found: {len(df)}")

df = df.head(1000)

# ==========================================
# 3. DIRECT PERSISTENT INGESTION LOOP
# ==========================================
print("Extracting vectors and executing force-commits...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_id = str(row['id'])
    img_path = row['image_path']
    
    rich_description = (
        f"A {row['baseColour']} {row['articleType']} designed for {row['gender']}s, "
        f"ideal for {row['usage']} wear. Title: {row['productDisplayName']}"
    )
    
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # ---  PURE PYTORCH LAYER EXTRACTION (NO WRAPPERS) ---
            vision_outputs = model.vision_model(**inputs)
            pooled_output = vision_outputs.pooler_output  # Pure 768-dim Tensor
            
            # Project to unified 512 multimodal space using the model's projection layer
            raw_features = model.visual_projection(pooled_output)  # Pure 512-dim Tensor
            
            # Math normalization works absolutely flawlessly now!
            image_features = raw_features / raw_features.norm(p=2, dim=-1, keepdim=True)
            vector_embedding = image_features.cpu().numpy().flatten().tolist()
        
        # Immediate database commit
        collection.add(
            embeddings=[vector_embedding],
            documents=[rich_description],
            metadatas=[{
                "image_path": str(img_path),
                "gender": str(row['gender']),
                "articleType": str(row['articleType']),
                "baseColour": str(row['baseColour'])
            }],
            ids=[img_id]
        )
        
    except Exception as item_error:
        print(f"\n Error adding product {img_id}: {item_error}")
        continue

print(f"\n FINAL VERIFICATION CHECK: Total items saved inside ChromaDB: {collection.count()}")