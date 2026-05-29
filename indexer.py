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
#  Strictly mapped to your current Windows repository folder structure
CSV_PATH = r"C:\Users\Lalisha\Desktop\Multimodal_fashion_discovery_Engine\fashion-dataset\styles.csv"
IMAGES_DIR = r"C:\Users\Lalisha\Desktop\Multimodal_fashion_discovery_Engine\fashion-dataset\images"

# Checking device type for embedding acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for embedding extraction: {device}")

# Loading Multimodal Architecture Model
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Initializing Native ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Cleaning any old corrupted collection reference
try:
    chroma_client.delete_collection(name="fashion_retrieval")
    print(" Old database collection wiped clean.")
except Exception:
    pass

# Creating fresh unified multimodal collection schema
collection = chroma_client.get_or_create_collection(
    name="fashion_retrieval", 
    metadata={"hnsw:space": "cosine"}
)

# ==========================================
# 2. METADATA LOADING & DATA VALIDATION
# ==========================================
print("Loading styles.csv metadata...")
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')

# Cleaning missing records
df = df.dropna(subset=['id', 'productDisplayName'])
df['id'] = df['id'].astype(int)

# Mapping strict relative layout matching paths
df['image_path'] = df['id'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))

# Filtering database entries to match only valid existing disk files
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
print(f"Total valid matching images found on disk: {len(df)}")

# Limiting to top 1000 items for micro-container stability on Cloud Environments
df = df.head(1000)

# ==========================================
# 3. MULTIMODAL FEATURE EXTRACTION & INGESTION
# ==========================================
print("Extracting visual vectors and executing absolute atomic indexing...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_id = str(row['id'])
    img_path = row['image_path']
    
    # Constructing descriptive prompt strings for alignment enrichment
    rich_description = (
        f"A {row['baseColour']} {row['articleType']} designed for {row['gender']}s, "
        f"ideal for {row['usage']} wear. Title: {row['productDisplayName']}"
    )
    
    try:
        # Image pre-processing
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Extracting representation layer features directly
            vision_outputs = model.vision_model(**inputs)
            pooled_output = vision_outputs.pooler_output  # Pure 768-dim baseline Tensor
            
            # Map features onto shared cross-modal vector target space (512-dim)
            raw_features = model.visual_projection(pooled_output)  
            
            # Mathematical standard normalization (Guarantees true cosine scaling matrix computation)
            image_features = raw_features / raw_features.norm(p=2, dim=-1, keepdim=True)
            vector_embedding = image_features.cpu().numpy().flatten().tolist()
        
        # Database commit phase
        collection.add(
            embeddings=[vector_embedding],
            documents=[rich_description],
            metadatas=[{
                "image_path": str(img_path),
                "gender": str(row['gender']),
                "articleType": str(row['articleType']),
                "baseColour": str(row['baseColour'])
            }],
            ids=[str(img_id)]  
        )
        
    except Exception as item_error:
        print(f"\n⚠️ Error adding product ID {img_id}: {item_error}")
        continue

# Final health and count reporting
print("\n" + "="*50)
print(f"🎉 SUCCESS: Indexing Pipeline Finalized!")
print(f" Total items securely saved inside local ChromaDB: {collection.count()}")
print("="*50)