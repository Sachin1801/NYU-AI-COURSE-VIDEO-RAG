# Chat with Your Video Library: A Retrieval-Augmented Generation (RAG) System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3118/)

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to allow users to query a video lecture library. The system retrieves the most relevant video segments—comprising frames and corresponding subtitles—and then generates answers to user queries based on those segments. This end-to-end solution enables interactive learning and information retrieval from video content.

![System Architecture](docs/system_architecture.png)

## Key Components

- **ETL Pipeline**: Ingests a video dataset (provided in webdataset format from Hugging Face), extracts frames and metadata (subtitles, timestamps), and stores the processed data in MongoDB.
  
- **Featurization & Embedding**: Converts subtitle text (and optionally image data) into embeddings using models such as `sentence-transformers` or a vision-language model.
  
- **Vector Storage & Retrieval**: Stores the embeddings in a vector database (e.g., Qdrant) and retrieves the top-K relevant segments based on a user's query.
  
- **LLM Integration & Prompting**: Constructs prompts using retrieved segments and generates detailed answers with a local or Hugging Face-based Large Language Model (LLM).
  
- **Deployment via Gradio**: Provides an interactive user interface with streaming answers and clickable video segments.

<!-- ## Directory Structure
```
my_project/
├── notebooks/
│   ├── 01_data_ingestion.ipynb        # ETL: loading webdataset and storing into MongoDB
│   ├── 02_embedding_retrieval.ipynb   # Embedding, vector storage, and retrieval demo
│   ├── 03_llm_integration.ipynb       # LLM prompt construction and answer generation
│   └── 04_gradio_app.ipynb            # Gradio interface for end-to-end demo
├── src/
│   ├── etl/
│   │   ├── download_and_ingest.py     # Code to load the webdataset and insert records into MongoDB
│   │   ├── process_sample.py          # Functions to decode samples and extract metadata
│   │   └── utils.py                   # Utility functions for ETL (logging, error handling, etc.)
│   ├── rag_pipeline/
│   │   ├── embedder.py                # Code for generating text and (optionally) image embeddings
│   │   ├── retriever.py               # Functions for querying the vector store (Qdrant)
│   │   ├── generator.py               # LLM integration code for generating responses
│   │   └── prompt_engineering.py      # Functions for constructing prompts from retrieved segments
│   └── deployment/
│       └── gradio_app.py              # Gradio app that ties together the entire pipeline
├── requirements.txt                   # Python dependencies for the project
├── README.md                          # This detailed README file
└── .gitignore                         # Git ignore settings
``` -->
<!-- 
## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Set Up a Virtual Environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   
   Create a `.env` file (or configure your environment) with the following variables:
   ```
   MONGO_URI=mongodb+srv://<username>:<password>@cluster0.xxxxxx.mongodb.net/?retryWrites=true&w=majority
   HF_DATASET_URL=hf://your_huggingface_dataset_path
   QDRANT_URI=http://localhost:6333  # or your Qdrant server URL
   OPENAI_API_KEY=your_openai_api_key  # or alternative LLM service
   ``` -->

## Detailed Pipeline Description

### 1. ETL Pipeline (Data Ingestion)

- **Source**: The dataset is provided in webdataset format via Hugging Face.

- **Processing Steps**:
  - **Load Webdataset**: Stream data using the webdataset library.
  - **Decode Samples**: Convert image bytes to PIL Images and parse JSON/text metadata.
  - **Extract Metadata**: Retrieve frame timestamps, subtitles, and video IDs.
  - **Insert into MongoDB**: Store each processed record with a unique index (based on video_id and frame_timestamp) to prevent duplicates.

- **Key Considerations**:
  - Validate the existence and correctness of metadata fields.
  - Use batching for large datasets to manage memory usage.
  - Handle errors and duplicates gracefully using MongoDB's unique indices.

### 2. Embedding & Vector Storage

- **Embedding**: Convert subtitle text (and optionally image data) into embeddings using:
  - **Text**: sentence-transformers (e.g., all-MiniLM-L6-v2)
  - **Image**: A vision-language model (optional, e.g., CLIP or BLIP)

- **Vector Store**: Store embeddings in Qdrant for fast similarity search.

- **Key Considerations**:
  - Ensure the embeddings are of consistent dimensionality.
  - Normalize embeddings if required.
  - Use proper indexing in Qdrant to optimize retrieval performance.

### 3. Retrieval Pipeline

- **Query Processing**: Convert user queries into embeddings and perform a similarity search on the vector store.

- **Return Top-K Results**: Fetch the most relevant video segments along with metadata (frame image path, subtitle text, timestamp, etc.).

- **Key Considerations**:
  - Tune the similarity threshold and the number of retrieved results.
  - Incorporate reranking or filtering mechanisms if necessary.

### 4. LLM Integration & Prompt Construction

- **Prompt Engineering**: Construct a detailed prompt combining:
  - A system message (e.g., "You are an AI tutor answering questions based on lecture videos.")
  - The user's query.
  - The concatenated text of the retrieved segments (subtitles) along with metadata references.

- **Response Generation**: Use a local or Hugging Face-based LLM to generate the final answer.

- **Key Considerations**:
  - Ensure the prompt does not exceed the LLM's token limit.
  - Validate that the generated response references the correct video segments.

### 5. Deployment via Gradio

- **Interface**: A Gradio app is provided for a seamless user experience.

- **Features**:
  - Text input for user queries.
  - Streaming LLM responses.
  - Display of retrieved video segments (with clickable timestamps for playback).

- **Key Considerations**:
  - Ensure the app is responsive and handles errors gracefully.
  - Test the app with a variety of queries to validate end-to-end performance.

## Steps to Run the Entire Pipeline

1. **ETL Pipeline (Data Ingestion)**
   - Run the notebook `notebooks/01_data_ingestion.ipynb` or execute the script in `src/etl/download_and_ingest.py` to load the webdataset and store processed data in MongoDB.

2. **Embedding and Retrieval**
   - Execute `notebooks/02_embedding_retrieval.ipynb` to convert text into embeddings and store them in Qdrant. Verify retrieval of top-K video segments for sample queries.

3. **LLM Integration**
   - Run `notebooks/03_llm_integration.ipynb` to test prompt construction and response generation using the LLM.

4. **Gradio App Deployment**
   - Finally, run the Gradio app via `notebooks/04_gradio_app.ipynb` or directly execute `src/deployment/gradio_app.py`:
     ```bash
     python src/deployment/gradio_app.py
     ```
   - Open the provided URL in your browser to interact with the system.

## Testing & Debugging

- **Unit Tests**: Add tests in a `tests/` directory to verify functions in ETL, embedding, retrieval, and LLM prompt construction.

- **Logging**: Use logging (or print statements during development) to monitor processing at each step.

- **Error Handling**: Ensure that network errors, missing metadata, and duplicate data insertions are gracefully managed.

## Collaboration & Version Control

- **GitHub Repository**: Push your changes frequently. Use branches for major features (ETL, embedding, LLM, UI).

- **Cursor Pro**: Utilize Cursor Pro for real-time code analysis, debugging, and collaboration. Clone your GitHub repository in Cursor, make changes, test locally, and then push updates.

## Acknowledgments

- **Dataset**: Provided by Pantelis Monogioudis/New York University.

- **Inspiration & References**:
  - The book "Chat with Your Video Library" for RAG pipeline examples.
  - Hugging Face WebDataset Documentation
  - MongoDB, Qdrant, and Gradio documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please contact Sachin Adlakha at sa9082@nyu.edu .