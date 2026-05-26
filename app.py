import streamlit as st
import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

st.set_page_config(page_title="AI Fashion Discovery Engine", layout="wide")

st.title("🛍️ Multimodal AI: Cross-Modal Fashion Discovery Engine")
st.write("Type semantic, mood-based, or conversational descriptions to scan the Vector Database in real-time.")

@st.cache_resource
def load_multimodal_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_NAME = "openai/clip-vit-base-patch32"
    
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name="fashion_retrieval")
    
    return model, processor, collection, device

model, processor, collection, device = load_multimodal_resources()

with st.sidebar:
    st.header("📊 Database Metrics")
    st.metric(label="Total Indexed Assets", value=f"{collection.count()} products")
    st.write("---")
    st.info("💡 **Try Queries Like:**\n"
            "- *Something formal to wear to a client meeting*\n"
            "- *Bright red casual summer dress*\n"
            "- *Sporty running shoes for athletic use*")

query_text = st.text_input(
    label="🔍 What specific vibe or outfit are you looking for today?", 
    placeholder="e.g., Casual black apparel for a cool winter evening walk..."
)

top_k = st.slider("Select maximum products to return (Top-K):", min_value=1, max_value=12, value=4)

if query_text:
    st.subheader(f"Results for: '{query_text}'")
    
    with st.spinner("Executing similarity search..."):
        with torch.no_grad():
            inputs = processor(text=[query_text], return_tensors="pt", padding=True).to(device)
            
            # --- 🛠️ THE FINAL CRITICAL FRONTEND FIX ---
            # Pure text model representation extraction layer (Guarantees no pixel_value error)
            text_outputs = model.text_model(**inputs)
            pooled_output = text_outputs.pooler_output  # Pure 768-dim Tensor
            
            # Project to unified 512 multimodal space using the model's text projection matrix
            raw_text_features = model.text_projection(pooled_output)  # Pure 512-dim Tensor
                
            # Perfect mathematical length normalization match
            text_features = raw_text_features / raw_text_features.norm(p=2, dim=-1, keepdim=True)
            query_vector = text_features.cpu().numpy().flatten().tolist()
        
        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k
            )
            
            if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                
                # --- 🛠️ THE GRID IMPROVEMENT FIX ---
                # Ek row mein maximum kitne columns chahiye (4 in a row)
                ITEMS_PER_ROW = 4
                
                # Loop chalayenge jo har bar 4 items ka batch uthayega
                for i in range(0, len(metadatas), ITEMS_PER_ROW):
                    # 4 items ka slice nikalenge
                    row_metadatas = metadatas[i:i + ITEMS_PER_ROW]
                    row_distances = distances[i:i + ITEMS_PER_ROW]
                    
                    # Create exactly 4 clean layout columns for this row
                    cols = st.columns(ITEMS_PER_ROW)
                    
                    for idx, meta in enumerate(row_metadatas):
                        with cols[idx]:
                            img_path = meta.get('image_path', '')
                            
                            if os.path.exists(img_path):
                                image = Image.open(img_path)
                                similarity_percentage = (1 - (row_distances[idx] / 2)) * 100
                                
                                # Visual card formatting using markdown borders
                                st.image(image, use_container_width=True)
                                st.markdown(f"**Match Quality:** `{similarity_percentage:.1f}%`")
                                st.caption(f"**Type:** {meta.get('articleType', 'N/A')}")
                                st.caption(f"**Color:** {meta.get('baseColour', 'N/A')}")
                            else:
                                st.error("Image file path mismatch.")
                    
                    # Thoda sa gap har row ke beech mein visual clearance ke liye
                    st.write("")
            else:
                st.warning("No matching products found.")
        except Exception as query_err:
            st.error(f"Execution Error Encountered: {query_err}")