import streamlit as strl
import requests

strl.set_page_config(
    page_title="Legal-Ease AI Dashboard",
    page_icon="⚖️",
    layout="wide"
)

BACKEND_API_URL = "http://127.0.0.1:8000"

strl.title("⚖️ Legal-Ease AI: Enterprise Document RAG Explorer")
strl.markdown("---")

left_column, right_column = strl.columns([1, 2])

with left_column:
    strl.header("📁 Data Ingestion Hub")
    strl.write("Upload structural PDFs to extract text segments and compile mathematical vector indexes.")
    
    uploaded_file = strl.file_uploader("Choose a PDF document file", type=["pdf"])
    
    if uploaded_file is not None:
        if strl.button("🚀 Process & Index Document", use_container_width=True):
            with strl.spinner("Parsing text segments and generating vector embeddings..."):
                try:
                    files_payload = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    endpoint_response = requests.post(f"{BACKEND_API_URL}/upload_doc", files=files_payload)
                    
                    if endpoint_response.status_code == 200:
                        json_data = endpoint_response.json()
                        if json_data.get("status") == "success":
                            strl.success(f"🎉 Success! Chunked and indexed {json_data.get('total_chunks_indexed')} sections.")
                        else:
                            strl.error(f"❌ Ingestion error: {json_data.get('detail')}")
                    else:
                        strl.error(f"❌ Server communication glitch: HTTP {endpoint_response.status_code}")
                except Exception as ex:
                    strl.error(f"⚠️ Unable to hook up with the backend network engine: {str(ex)}")

with right_column:
    strl.header("💬 AI Analytical Query Interface")
    
    if "chat_history" not in strl.session_state:
        strl.session_state.chat_history = []
        
    for chat_entry in strl.session_state.chat_history:
        with strl.chat_message(chat_entry["role"]):
            strl.write(chat_entry["content"])
            
    user_prompt_input = strl.chat_input("Ask a contextual question about your uploaded legal/financial files...")
    
    if user_prompt_input:
        with strl.chat_message("user"):
            strl.write(user_prompt_input)
        strl.session_state.chat_history.append({"role": "user", "content": user_prompt_input})
        
        with strl.chat_message("assistant"):
            with strl.spinner("Evaluating vector similarity space and fetching answer..."):
                try:
                    query_payload = {"prompt": user_prompt_input, "num_results": 3}
                    endpoint_response = requests.post(f"{BACKEND_API_URL}/query", json=query_payload)
                    
                    if endpoint_response.status_code == 200:
                        output_data = endpoint_response.json()
                        ai_answer = output_data.get("ai_response", "No readable reply generated.")
                        
                        strl.write(ai_answer)
                        strl.session_state.chat_history.append({"role": "assistant", "content": ai_answer})
                        
                        with strl.expander("🔍 Inspect Ground-Truth Context Blocks (ChromaDB)"):
                            retrieved_text_fragments = output_data.get("retrieved_context", [])
                            if retrieved_text_fragments:
                                for iterator, chunk_text in enumerate(retrieved_text_fragments):
                                    strl.info(f"**Chunk [{iterator + 1}]:** {chunk_text}")
                            else:
                                strl.warning("No context fragments were retrieved for this query.")
                    else:
                        strl.error(f"❌ Backend endpoint dropped transaction request: HTTP {endpoint_response.status_code}")
                except Exception as ex:
                    strl.error(f"⚠️ App pipeline transmission breakdown: {str(ex)}")
