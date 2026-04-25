from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="Astro-LLaMA API")

# Enable CORS (Allows your React app at localhost:3000 to talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

class AstroRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_abstract(request: AstroRequest):
    # Ollama's local API URL
    OLLAMA_URL = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "astro-llama",
        "prompt": request.prompt,
        "stream": False # Set to True if you want to implement typing-effects in React
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            return {"completion": data.get("response", "")}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)