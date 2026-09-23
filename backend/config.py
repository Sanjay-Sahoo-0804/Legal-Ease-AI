import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
_llm_pipeline = None

def get_llm_pipeline():
    global _llm_pipeline
    if _llm_pipeline is not None:
        return _llm_pipeline

    print(f"🤖 Loading Hugging Face LLM Model: {MODEL_ID}...")
    
    if torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.float16
        print("✅ Using GPU acceleration.")
    else:
        device_map = None
        torch_dtype = torch.float32
        print("⚠️ Falling back to CPU processing.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager"  # Bypasses Triton compiler issue completely
        )
        
        _llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.9
        )
        print("🎉 LLM Pipeline loaded successfully.")
        return _llm_pipeline
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        raise e

def generate_llm_response(question: str, contexts: list) -> str:
    combined_context = "\n\n".join([f"Context:\n{c}" for c in contexts])
    
    system_instruction = (
        "Answer the question accurately using ONLY the provided contexts snippets. "
        "If you don't know the answer, say 'I cannot find the answer within the provided documents.' "
        "Do not invent any facts."
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Context snippets:\n{combined_context}\n\nQuestion: {question}"}
    ]
    
    try:
        pipe = get_llm_pipeline()
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt)
        generated_text = outputs[0]["generated_text"]
        return generated_text.split(prompt)[-1].strip()
    except Exception as e:
        return f"Model inference execution exception: {str(e)}"
