import streamlit as st
import torch
import clip
from PIL import Image
import os
import numpy as np
import chromadb
import requests
import tempfile
import time

# ----- Setup -----
CACHE_DIR = tempfile.gettempdir()
CHROMA_PATH = os.path.join(CACHE_DIR, "chroma_db")
DEMO_DIR = os.path.join(CACHE_DIR, "demo_images")
os.makedirs(DEMO_DIR, exist_ok=True)

# ----- Initialize Session State -----
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None
if 'demo_images' not in st.session_state:
    st.session_state.demo_images = []
if 'user_images' not in st.session_state:
    st.session_state.user_images = []

# ----- Load CLIP Model -----
if 'model' not in st.session_state:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=CACHE_DIR)
    st.session_state.model = model
    st.session_state.preprocess = preprocess
    st.session_state.device = device

# ----- Initialize ChromaDB -----
if 'chroma_client' not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    st.session_state.demo_collection = st.session_state.chroma_client.get_or_create_collection(
        name="demo_images", metadata={"hnsw:space": "cosine"}
    )
    st.session_state.user_collection = st.session_state.chroma_client.get_or_create_collection(
        name="user_images", metadata={"hnsw:space": "cosine"}
    )

# ----- Title -----
st.title("🔍 CLIP-Based Image Search")

# ----- Dataset Buttons -----
col1, col2 = st.columns(2)
if col1.button("📦 Use Demo Images"):
    st.session_state.dataset_name = "demo"
    st.session_state.dataset_loaded = False

if col2.button("📤 Upload Your Images"):
    st.session_state.dataset_name = "user"
    st.session_state.dataset_loaded = False

# ----- Download + Embed Demo Images -----
def download_image_with_retry(url, path, retries=3, delay=1.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    f.write(r.content)
                return True
        except Exception as e:
            time.sleep(delay)
    return False

if st.session_state.dataset_name == "demo" and not st.session_state.dataset_loaded:
    with st.spinner("Downloading and indexing demo images..."):
        st.session_state.demo_collection.delete(ids=[str(i) for i in range(50)])
        demo_image_paths, demo_images = [], []
        for i in range(50):
            path = os.path.join(DEMO_DIR, f"img_{i+1:02}.jpg")
            if not os.path.exists(path):
                url = f"https://picsum.photos/seed/{i}/1024/768"
                download_image_with_retry(url, path)
            try:
                demo_images.append(Image.open(path).convert("RGB"))
                demo_image_paths.append(path)
            except:
                continue  # skip corrupted

        embeddings, ids, metadatas = [], [], []
        for i, img in enumerate(demo_images):
            img_tensor = st.session_state.preprocess(img).unsqueeze(0).to(st.session_state.device)
            with torch.no_grad():
                embedding = st.session_state.model.encode_image(img_tensor).cpu().numpy().flatten()
            embeddings.append(embedding)
            ids.append(str(i))
            metadatas.append({"path": demo_image_paths[i]})

        st.session_state.demo_collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)
        st.session_state.demo_images = demo_images
        st.session_state.dataset_loaded = True

    st.success("✅ Demo images loaded!")

# ----- Upload User Images -----
if st.session_state.dataset_name == "user" and not st.session_state.dataset_loaded:
    uploaded = st.file_uploader("Upload your images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded:
        st.session_state.user_collection.delete(ids=[
            str(i) for i in range(st.session_state.user_collection.count())
        ])
        user_images = []
        for i, file in enumerate(uploaded):
            try:
                img = Image.open(file).convert("RGB")
            except:
                continue
            user_images.append(img)
            img_tensor = st.session_state.preprocess(img).unsqueeze(0).to(st.session_state.device)
            with torch.no_grad():
                embedding = st.session_state.model.encode_image(img_tensor).cpu().numpy().flatten()
            st.session_state.user_collection.add(
                embeddings=[embedding], ids=[str(i)], metadatas=[{"index": i}]
            )

        st.session_state.user_images = user_images
        st.session_state.dataset_loaded = True
        st.success(f"✅ Uploaded {len(user_images)} images.")

# ----- Search Section -----
if st.session_state.dataset_loaded:
    st.subheader("🔎 Search Section")
    query_type = st.radio("Search by:", ("Text", "Image"))

    query_embedding = None
    if query_type == "Text":
        text_query = st.text_input("Enter search text:")
        if text_query:
            tokens = clip.tokenize([text_query]).to(st.session_state.device)
            with torch.no_grad():
                query_embedding = st.session_state.model.encode_text(tokens).cpu().numpy().flatten()

    elif query_type == "Image":
        query_file = st.file_uploader("Upload query image", type=["jpg", "jpeg", "png"], key="query_image")
        if query_file:
            query_img = Image.open(query_file).convert("RGB")
            st.image(query_img, caption="Query Image", width=200)
            query_tensor = st.session_state.preprocess(query_img).unsqueeze(0).to(st.session_state.device)
            with torch.no_grad():
                query_embedding = st.session_state.model.encode_image(query_tensor).cpu().numpy().flatten()

    # ----- Perform Search -----
    if query_embedding is not None:
        if st.session_state.dataset_name == "demo":
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

            st.subheader("🔗 Top Matches")
            cols = st.columns(len(ids))
            for i, (img_id, sim) in enumerate(zip(ids, similarities)):
                with cols[i]:
                    st.image(images[int(img_id)], caption=f"Sim: {sim:.3f}", width=150)
        else:
            st.warning("No indexed images to search.")
else:
    st.info("👆 Please select a dataset (Demo or Upload Images) to begin.")
