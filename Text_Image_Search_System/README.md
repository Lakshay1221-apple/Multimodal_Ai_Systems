# Text-to-Image Search System

This project demonstrates a simple semantic image search system using CLIP. A user enters a text query, the notebook converts that query into a text embedding, compares it with image embeddings from the local `Images` folder, and returns the image that best matches the meaning of the query.

## Project Structure

```text
Text_Image_Search_System/
|-- Images/
|   |-- Cat.jpg
|   |-- Dog.jpg
|   |-- Elephant.jpg
|   |-- aeroplane.jpg
|   |-- bird.jpeg
|   |-- building.jpg
|   |-- horse.jpg
|   |-- laptop.jpeg
|   |-- mountain.jpeg
|   `-- pizza.jpg
|-- main.ipynb
`-- README.md
```

## What the Program Does

The notebook builds a small retrieval pipeline with the `openai/clip-vit-base-patch32` model from Hugging Face Transformers. CLIP is trained to represent text and images in a shared embedding space, which makes it possible to compare a sentence such as `Horse` with a folder of images.

The program follows these steps:

1. Import the required libraries: `transformers`, `PIL`, `torch`, and `os`.
2. Load the pretrained CLIP model and processor.
3. Read every image from the `Images` directory.
4. Convert each image into a normalized image embedding.
5. Stack all image embeddings into a searchable tensor.
6. Convert a user text query into a normalized text embedding.
7. Compute similarity scores between the text embedding and all image embeddings.
8. Return the highest-scoring image path.
9. Optionally display the top-k matching images.

## Requirements

Install the required Python packages before running the notebook:

```bash
pip install torch transformers pillow ipython jupyter
```

If you are using a virtual environment, activate it before installing the packages.

## How to Run

1. Open `main.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
2. Run the cells from top to bottom.
3. Wait for the CLIP model and processor to load.
4. Confirm that the images from the `Images` folder are embedded successfully.
5. Change the query value, for example:

```python
result = search("Horse")
print(result)
```

6. Run the top-k result cells to display multiple matching images.

## Important Path Note

The notebook currently uses an absolute path for `image_dir`:

```python
image_dir = "/home/lakshay/Multimodal_Embeddings/Text_Image_Search_System/Images"
```

If you move the project to another machine or folder, update this path or replace it with a relative path:

```python
image_dir = "Images"
```

## Core Search Logic

The search function embeds the text query and compares it with all image embeddings:

```python
scores = torch.matmul(text_features, image_embedding.T)
best_index = scores.argmax().item()
return image_paths[best_index]
```

Because the embeddings are normalized first, the matrix multiplication acts like cosine similarity. The image with the highest score is treated as the best semantic match.

## Example Queries

Try queries that describe the image content:

```text
Horse
Dog
Pizza
Mountain
Aeroplane
Laptop
```

You can also try more descriptive prompts, such as:

```text
a domestic animal
a flying vehicle
a food item
a natural landscape
```

## Output

The notebook can produce two types of results:

- A single best-matching image path using the `search(query)` function.
- The top three ranked images using `torch.topk`.

## Summary

This project shows how multimodal embeddings can connect natural language and images. Even with a small image folder, the same idea can scale into larger search systems by embedding more images and storing those vectors in a vector database or search index.
