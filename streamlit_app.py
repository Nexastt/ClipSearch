import streamlit as st
import torch
import clip
from PIL import Image
import os
import numpy as np
import chromadb
import tempfile

# ----- Setup -----
CACHE_DIR = tempfile.gettempdir()
CHROMA_PATH = os.path.join(CACHE_DIR, "chroma_db")

# ----- Load CLIP Model -----
if 'model' not in st.session_state:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=CACHE_DIR)
    st.session_state.model = model
    st.session_state.preprocess = preprocess
    st.session_state.device = device
    st.session_state.demo_images = []
    st.session_state.demo_image_paths = []
    st.session_state.user_images = []

# ----- Initialize ChromaDB -----
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    st.session_state.demo_collection = st.session_state.chroma_client.get_or_create_collection(
        name="demo_images", metadata={"hnsw:space": "cosine"}
    )
    st.session_state.user_collection = st.session_state.chroma_client.get_or_create_collection(
        name="user_images", metadata={"hnsw:space": "cosine"}
    )

# ----- Load Demo Images -----
if not st.session_state.get("demo_images_loaded", False):
    demo_folder = "demo_images"
    if os.path.exists(demo_folder):
        demo_image_paths = [os.path.join(demo_folder, f) for f in os.listdir(demo_folder)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        st.session_state.demo_images = [Image.open(p).convert("RGB") for p in demo_image_paths]
        st.session_state.demo_image_paths = demo_image_paths

        st.session_state.demo_collection.delete(ids=[str(i) for i in range(len(demo_image_paths))])

        embeddings, ids, metadatas = [], [], []
        for i, img in enumerate(st.session_state.demo_images):
            img_tensor = st.session_state.preprocess(img).unsqueeze(0).to(st.session_state.device)
            with torch.no_grad():
                embedding = st.session_state.model.encode_image(img_tensor).cpu().numpy().flatten()
            embeddings.append(embedding)
            ids.append(str(i))
            metadatas.append({"path": demo_image_paths[i]})

        st.session_state.demo_collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)
        st.session_state.demo_images_loaded = True

# ----- UI -----
st.title("🔎 CLIP Image Search (Text & Image)")
mode = st.radio("Choose dataset to search in:", ("Demo Images", "My Uploaded Images"))
query_type = st.radio("Query type:", ("Image", "Text"))

# ----- Upload User Images -----
if mode == "My Uploaded Images":
    uploaded = st.file_uploader("Upload your images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    if uploaded:
        st.session_state.user_images = []
        st.session_state.user_collection.delete(ids=[
            str(i) for i in range(st.session_state.user_collection.count())
        ])

        for i, file in enumerate(uploaded):
            img = Image.open(file).convert("RGB")
            st.session_state.user_images.append(img)

            img_tensor = st.session_state.preprocess(img).unsqueeze(0).to(st.session_state.device)
            with torch.no_grad():
                embedding = st.session_state.model.encode_image(img_tensor).cpu().numpy().flatten()

            st.session_state.user_collection.add(
                embeddings=[embedding],
                ids=[str(i)],
                metadatas=[{"index": i}]
            )

        st.success(f"{len(uploaded)} images uploaded.")

# ----- Perform Query -----
query_embedding = None
if query_type == "Image":
    img_file = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"])
    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, caption="Query Image", width=200)
        img_tensor = st.session_state.preprocess(img).unsqueeze(0).to(st.session_state.device)
        with torch.no_grad():
            query_embedding = st.session_state.model.encode_image(img_tensor).cpu().numpy().flatten()
elif query_type == "Text":
    text_query = st.text_input("Enter search text:")
    if text_query:
        tokens = clip.tokenize([text_query]).to(st.session_state.device)
        with torch.no_grad():
            query_embedding = st.session_state.model.encode_text(tokens).cpu().numpy().flatten()

# ----- Run Search -----
if query_embedding is not None:
    if mode == "Demo Images":
        collection = st.session_state.demo_collection
        images = st.session_state.demo_images
    else:
        collection = st.session_state.user_collection
        images = st.session_state.user_images

    if collection.count() > 0:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(5, collection.count())
        )
        ids = results["ids"][0]
        distances = results["distances"][0]
        similarities = [1 - d for d in distances]

        st.subheader("Top Matches")
        cols = st.columns(5)
        for i, (img_id, sim) in enumerate(zip(ids, similarities)):
            with cols[i]:
                idx = int(img_id)
                st.image(images[idx], caption=f"Sim: {sim:.3f}", width=150)
    else:
        st.warning("No images found in collection.")
