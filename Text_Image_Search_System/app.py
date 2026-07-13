from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_DIR / "Images"
MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


st.set_page_config(
    page_title="Text Image Search",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading CLIP model...")
def load_clip():
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


@st.cache_resource(show_spinner="Indexing image library...")
def build_image_index():
    model, processor = load_clip()
    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        return torch.empty(0), []

    embeddings = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model.vision_model(**inputs)
            features = model.visual_projection(outputs.pooler_output)

        features = features / features.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(features)

    return torch.cat(embeddings, dim=0), image_paths


def search_images(query, top_k):
    model, processor = load_clip()
    image_embeddings, image_paths = build_image_index()

    if image_embeddings.numel() == 0:
        return []

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.text_model(**inputs)
        text_features = model.text_projection(outputs.pooler_output)

    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    scores = torch.matmul(text_features, image_embeddings.T).squeeze(0)
    top_k = min(top_k, len(image_paths))
    values, indices = torch.topk(scores, k=top_k)

    return [
        {
            "path": image_paths[index.item()],
            "name": image_paths[index.item()].stem.replace("_", " ").title(),
            "score": value.item(),
        }
        for value, index in zip(values, indices)
    ]


st.markdown(
    """
    <style>
    .stApp {
        background: #f7f8fb;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    .subtitle {
        color: #4b5563;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.85rem;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        min-height: 100%;
    }
    .rank {
        color: #6b7280;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .result-name {
        color: #111827;
        font-size: 1.15rem;
        font-weight: 750;
        margin-top: 0.2rem;
    }
    .score {
        color: #2563eb;
        font-size: 0.9rem;
        font-weight: 650;
        margin-bottom: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Search Settings")
    image_embeddings, image_paths = build_image_index()
    result_count = st.slider(
        "Number of results",
        min_value=1,
        max_value=max(1, len(image_paths)),
        value=min(3, max(1, len(image_paths))),
    )
    st.metric("Indexed images", len(image_paths))
    st.caption("Powered by CLIP semantic embeddings.")


st.markdown('<div class="main-title">Text to Image Search</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Type a natural-language query and explore the closest images from your local library.</div>',
    unsafe_allow_html=True,
)

query = st.text_input(
    "Enter Query",
    value="a cute animal",
    placeholder="Try: dog, flying vehicle, food item, natural landscape",
)

search_clicked = st.button("Search", type="primary", width="content")

if not image_paths:
    st.error(f"No images found in {IMAGE_DIR}")
elif query.strip() and (search_clicked or query):
    with st.spinner("Finding the best visual matches..."):
        results = search_images(query.strip(), result_count)

    st.subheader("Top Results")
    columns = st.columns(min(3, len(results)))

    for rank, result in enumerate(results, start=1):
        column = columns[(rank - 1) % len(columns)]
        with column:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="rank">Rank {rank}</div>', unsafe_allow_html=True)
            st.image(str(result["path"]), width="stretch")
            st.markdown(
                f'<div class="result-name">{result["name"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="score">Similarity: {result["score"]:.3f}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Enter a query to begin searching.")
