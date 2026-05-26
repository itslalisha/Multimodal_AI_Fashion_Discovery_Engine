#  Multimodal AI: Cross-Modal Fashion Discovery Engine

A production-ready **Semantic and Mood-Based Fashion Discovery Engine** built using OpenAI's **CLIP** model and **ChromaDB**. This system moves beyond traditional keyword/tag matching to understand the semantic genre, context, and visual aesthetics of user queries to retrieve relevant fashion assets in real-time.

---

##  Project Overview

Traditional e-commerce search engines rely on strict keyword tags, often failing when users search using conversational descriptions or abstract moods (e.g., *"cool outfit for a beach party"*). 

This project solves that limitation by implementing **Semantic Cross-Modal Retrieval**:
* Maps both unstructured text queries and images into a single **shared embedding space**.
* Processes text and image features into deterministic **512-dimensional vectors**.
* Utilizes **Cosine Similarity** to match the geometric directional closeness of text queries to indexed image coordinates.

---
##  Research Background & References

This implementation is strongly grounded in foundational machine learning and deep learning literature. The exact academic papers utilized for understanding the cross-attention mechanisms, multimodal alignment, and vector space efficiency are archived in the `/research_papers_used` directory:

* **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** - *OpenAI* (Explains the contrastive pre-training framework for matching text and image modalities).
* **Attention Is All You Need (Transformers)** - (Foundational paper for understanding the underlying self-attention layers within the CLIP Text and Vision Encoders).

---
##  Tech Stack & Tools Used

* **Core Framework:** PyTorch (`torch`) - For tensor computations and feature vector $L_2$ normalization.
* **Deep Learning Architecture:** Hugging Face Transformers (`CLIPModel`, `CLIPProcessor`) - Utilizing `openai/clip-vit-base-patch32`.
* **Vector Database:** ChromaDB - For high-performance persistent vector storage and Approximate Nearest Neighbor (ANN) search.
* **UI Dashboard:** Streamlit - For building a highly responsive, real-time e-commerce user interface with customized multi-row grids.
* **Data Processing:** Pandas & Pillow (PIL) - For structured metadata cleaning and image tensor transformations.
* **Monitoring:** tqdm - For live processing tracking during batch ingestion.

---

##  System Architecture & Workflow

1. **Data Ingestion & Cleaning (`indexer.py`):** * Reads metadata descriptors and checks local disk availability for image assets.
   * Passes raw images through the CLIP Vision Encoder to extract raw features.
   * Projects and normalizes vectors into stable 512-dimensional float arrays.
   * Persistent commits embeddings alongside structured metadata (Type, Color, Path) into ChromaDB.

2. **Real-Time Similarity Search (`app.py`):**
   * Accepts natural, conversational user queries via the Streamlit web layout.
   * Encodes text using the CLIP Text Encoder and maps it to the 512 multimodal space.
   * Queries ChromaDB using Cosine Distance metrics to isolate top-K matches.
   * Dynamic frontend loops structure results into a clean, professional 4-column electronic catalog grid.

---

##  Installation & Usage Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/NEW_Project_multiModel.git](https://github.com/your-username/NEW_Project_multiModel.git)
cd NEW_Project_multiModel