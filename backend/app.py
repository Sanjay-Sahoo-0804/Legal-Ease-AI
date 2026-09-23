import os
os.environ["TRITON_DISABLE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TRITON_DISABLE"] = "1"

import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Force Python to look at the root directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import generate_llm_response
from data_pipeline.preprocess import extract_text_from_pdf, clean_text, chunk_text
from backend.vector_store import add_chunks_to_vector_db, query_vector_db

app = FastAPI(
    title="Legal-Ease AI Core Backend API",
    version="1.0.0"
)

# Enable complete CORS communication rules
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    prompt: str
    num_results: int = 3

@app.get("/")
def read_root():
    return {"status": "online"}

@app.post("/upload_doc")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    import shutil
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        raw_text = extract_text_from_pdf(temp_file_path)
        if not raw_text.strip():
            raise HTTPException(status_code=422, detail="Empty PDF.")
            
        cleaned_text = clean_text(raw_text)
        text_chunks = chunk_text(cleaned_text, chunk_size=500, chunk_overlap=50)
        
        add_chunks_to_vector_db(text_chunks, source_name=file.filename)
        
        return {
            "status": "success",
            "filename": file.filename,
            "total_chunks_indexed": len(text_chunks)
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    try:
        retrieved_contexts = query_vector_db(user_query=request.prompt, num_results=request.num_results)
        
        # Safe extraction layer for diverse ChromaDB formats
        context_list = []
        if isinstance(retrieved_contexts, list):
            context_list = retrieved_contexts
        elif isinstance(retrieved_contexts, dict) and "documents" in retrieved_contexts:
            context_list = retrieved_contexts["documents"][0] if retrieved_contexts["documents"] else []
        
        ai_response = generate_llm_response(question=request.prompt, contexts=context_list)
        
        return {
            "query": request.prompt,
            "retrieved_context": context_list,
            "ai_response": ai_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
