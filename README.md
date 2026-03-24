# FRESH: Food Recommendation and Evaluation System for Health

FRESH is an intelligent food recommendation and evaluation system that leverages deep learning, retrieval-augmented generation (RAG), and symbolic constraint optimization to provide personalized, health-aligned recipe recommendations. It features a modular architecture with a FastAPI backend, Streamlit dashboard, and robust data processing and evaluation pipelines.

## Main Components

- **main.py**: FastAPI backend, core logic, and constraint penalty system.
- **app.py**: Streamlit web dashboard for user interaction.
- **model.py**: PyTorch model architecture (FRESH_Network).
- **nutrition_agent.py**: Nutrition analysis agent for recipe parsing and health alignment.
- **FRESH/recommender.py**: Recommendation engine logic.
- **preprocess.py**: Data cleaning and preparation.
- **generate_embeddings.py**: Embedding generation for recipes.
- **build_faiss.py**: FAISS index creation for fast retrieval.
- **train_synthesis.py**: Model training script.
- **evaluate_model.py**: Model evaluation suite.

## Setup

1. **Clone the repository**
2. **Install dependencies**
   - Using pip:
     ```bash
     pip install -r Requirements/test_pip_env.txt
     ```
   - Or using conda:
     ```bash
     conda env create -f Requirements/test_env.yaml
     conda activate test
     ```
3. **Run the backend**
   ```bash
   python main.py
   ```
4. **Launch the dashboard**
   ```bash
   streamlit run app.py
   ```

## Directory Structure

- `main.py`, `app.py`, `model.py`, `nutrition_agent.py`, `preprocess.py`, `generate_embeddings.py`, `build_faiss.py`, `train_synthesis.py`, `evaluate_model.py`, `agent.py`
- `FRESH/recommender.py`, `FRESH/streamlit_app.py`
- `Requirements/test_pip_env.txt`, `Requirements/test_env.yaml`

## Notes
- Data files, model checkpoints, and intermediate results are **not** included in the minimal repo push.
- For full functionality, ensure you have the required data and model files as described in the scripts.

## License
See individual model folders for license details.
