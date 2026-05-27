# 🚨 PROTOBUF COMPATIBILITY BYPASS (MUST BE AT THE VERY TOP)
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import chromadb
import torch
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="AI Fashion Discovery Engine", layout="wide")

st.title("🛍️ Multimodal AI: Cross-Modal Fashion Discovery Engine")
st.write("Type semantic, mood-based, or conversational descriptions to scan the Vector Database in real-time.")

@st.cache_resource
def load_multimodal_resources():
    # Cloud execution runs best on deterministic CPU to avoid out-of-memory crashes
    device = torch.device("cpu")
    MODEL_NAME = "openai/clip-vit-base-patch32"
    
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Cloud Database Fallback: Prevents app crash if collection registry is delayed
    try:
        collection = chroma_client.get_collection(name="fashion_retrieval")
    except Exception:
        collection = chroma_client.get_or_create_collection(
            name="fashion_retrieval", 
            metadata={"hnsw:space": "cosine"}
        )
        # Mocking a single element to hold the cloud database schema state stable
        collection.add(
            embeddings=[[0.0] * 512],
            documents=["A placeholder asset for database schema stabilization."],
            metadatas=[{"articleType": "Mock", "baseColour": "Mock"}],
            ids=["placeholder_id"]
        )
    
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
            
            # Pure text model representation extraction layer
            text_outputs = model.text_model(**inputs)
            pooled_output = text_outputs.pooler_output  
            
            # Project to unified 512 multimodal space
            raw_text_features = model.text_projection(pooled_output)  
                
            # Perfect mathematical length normalization match
            text_features = raw_text_features / raw_text_features.norm(p=2, dim=-1, keepdim=True)
            query_vector = text_features.cpu().numpy().flatten().tolist()
        
        try:
            # Safe limit to prevent requesting more items than available
            safe_top_k = max(1, min(top_k, collection.count()))
            
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=safe_top_k
            )
            
            if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                ITEMS_PER_ROW = 4
                for i in range(0, len(metadatas), ITEMS_PER_ROW):
                    row_metadatas = metadatas[i:i + ITEMS_PER_ROW]
                    row_distances = distances[i:i + ITEMS_PER_ROW]
                    row_ids = ids[i:i + ITEMS_PER_ROW]
                    
                    cols = st.columns(ITEMS_PER_ROW)
                    for idx, meta in enumerate(row_metadatas):
                        if idx < len(cols):
                            with cols[idx]:
                                similarity_percentage = (1 - (row_distances[idx] / 2)) * 100
                                
                                # --- 🛠️ THE CLOUD IMAGE FIX ---
                                # Replaced local C: Drive path with dynamic structural placeholder
                                st.image("https://placehold.co/300x400/png?text=Fashion+Item+" + str(row_ids[idx]), use_container_width=True)
                                
                                st.markdown(f"**Match Quality:** `{similarity_percentage:.1f}%`")
                                st.caption(f"**Type:** {meta.get('articleType', 'N/A')}")
                                st.caption(f"**Color:** {meta.get('baseColour', 'N/A')}")
                    st.write("")
            else:
                st.warning("No matching products found.")
        except Exception as query_err:
            st.error(f"Execution Error Encountered: {query_err}")