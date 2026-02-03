
import os
import json
import faiss
import logging
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from sentence_transformers import SentenceTransformer

# --- NEW IMPORTS (From your t2.py/t3.py scripts) ---
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# --- CONFIG ---
PDF_DIR = "/data1/home/sathvik/Documents/FRESH/daata/nutrition_pdfs"
OUTPUT_INDEX = "/data1/home/sathvik/Documents/FRESH/daata/nutrition.index"
OUTPUT_META = "/data1/home/sathvik/Documents/FRESH/daata/nutrition_chunks.json"


embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_docling_converter():
    """Sets up Docling with OCR and Table support"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE 
    
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

def process_pdfs_with_structure():
    if not os.path.exists(PDF_DIR):
        print(f"⚠️ {PDF_DIR} not found.")
        return []

    converter = get_docling_converter()
    final_chunks = []
    
    # 1. DEFINE HEADERS TO SPLIT ON (Logic from t3.py)
    # This tells the system: "When you see a big header, that's a new section."
    headers_to_split_on = [
        ("#", "Header_1"),      # e.g. "Chapter 1: Vitamins"
        ("##", "Header_2"),     # e.g. "Vitamin A"
        ("###", "Header_3"),    # e.g. "Deficiency Symptoms"
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # 2. DEFINE RECURSIVE SPLITTER (Logic from t2.py)
    # If a section is still too big (like a massive table), chop it down safely.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"🚀 Processing {len(files)} PDFs with Structure-Aware Splitting...")

    for filename in files:
        file_path = os.path.join(PDF_DIR, filename)
        try:
            print(f"   📖 Converting {filename} to Markdown...")
            conv_result = converter.convert(file_path)
            markdown_text = conv_result.document.export_to_markdown()

            # STEP A: Split by Headers (The "Context" Step)
            # This creates chunks that carry their parent headers as metadata!
            header_splits = markdown_splitter.split_text(markdown_text)

            # STEP B: Recursive Split (The "Safety" Step)
            # Ensures no chunk is too large for the embedding model
            final_splits = text_splitter.split_documents(header_splits)

            # STEP C: Format for Indexing
            for split in final_splits:
                # Combine headers into the content for better retrieval
                # Example: "Vitamin A > Deficiency: Night blindness is a symptom..."
                header_context = " > ".join([v for k,v in split.metadata.items() if k.startswith("Header")])
                
                rich_content = f"[{header_context}] {split.page_content}" if header_context else split.page_content

                final_chunks.append({
                    "source": filename,
                    "content": rich_content, 
                    "metadata": split.metadata # Saves {'Header_1': '...', 'Header_2': '...'}
                })
            
            print(f"   ✅ Parsed {filename} into {len(final_splits)} structured chunks.")

        except Exception as e:
            print(f"   ❌ Error on {filename}: {e}")

    return final_chunks

def build_index():
    docs = process_pdfs_with_structure()
    if not docs: return

    print(f"🔢 Embedding {len(docs)} structured chunks...")
    texts = [d['content'] for d in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    faiss.write_index(index, OUTPUT_INDEX)
    with open(OUTPUT_META, 'w') as f:
        json.dump(docs, f)
        
    print(f"🎉 Knowledge Base built! Saved to {OUTPUT_INDEX}")

if __name__ == "__main__":
    # logging.getLogger("docling").setLevel(logging.WARNING)
    build_index()