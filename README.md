# 🌌 Astro-LLaMA: Specialized Astronomy Abstract Completion

Astro-LLaMA is a fine-tuned Llama 3.1 8B model specialized for the domain of astrophysics. Using **QLoRA** and the **Unsloth** framework, the model was trained to "unlearn" its conversational assistant persona and adopt the formal, technical style of arXiv scientific abstracts.

---

## 🚀 The Stack
- **Base Model:** Llama 3.1 8B (Instruct)
- **Fine-Tuning:** QLoRA (4-bit quantization) via Unsloth
- **Dataset:** 500+ real scientific abstracts from UniverseTBD/arxiv-abstracts-large
- **Evaluation:** LLM-as-a-Judge (Gemini 2.5 Flash) vs Base Llama 3.1
- **Backend:** FastAPI + Ollama
- **Frontend:** React (Vite) + Tailwind CSS v4

---

## 🛠️ Technical Specifications & Architecture

### Model Architecture
The model uses a **Low-Rank Adaptation (LoRA)** approach. Instead of updating all 8 billion parameters, we inject trainable rank-decomposition matrices into the Transformer layers.
- **Rank (r):** 16
- **Alpha:** 16
- **Target Modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Precision:** 4-bit NormalFloat (NF4) for base weights, BFloat16 for adapters.

### Training Pipeline
1. **Data Prep:** Abstracts were split into "Introduction" (Input) and "Conclusion/Results" (Output).
2. **Fine-Tuning:** Performed on an NVIDIA A100 (80GB).
3. **Quantization:** Merged LoRA adapters and exported to **4-bit GGUF (Q4_K_M)** for high-performance local inference.



---

## ⚖️ Evaluation (AstroBench)
To measure success, we performed a side-by-side comparison. The fine-tuned model was judged on its ability to maintain scientific tone without "chatbot" pleasantries.
- **Base Model:** Frequently added "Sure, here is an abstract..." or markdown headers.
- **Astro-LLaMA:** Seamlessly continued the scientific text.
- **Final Result:** **2-0 Sweep** (Judged by Gemini 2.5 Flash).

---

## 💻 Local Setup & Execution

### 1. Prerequisites
- **Ollama** installed on Windows/Linux.
- **Python 3.10+**
- **Node.js & npm**
- **NVIDIA GPU** (Optimized for 5070 Ti or similar).

### 2. Model Registration
Place your .gguf and Modelfile in the /model folder, then run:
\\\powershell
ollama create astro-llama -f ./model/Modelfile
\\\

### 3. Backend Setup
\\\powershell
pip install fastapi uvicorn httpx
python server.py
\\\

### 4. Frontend Setup
\\\powershell
cd astro-frontend
npm install
npm run dev
\\\

---

## 🌌 Use Case
This model serves as a "Co-Pilot" for researchers. By inputting a preliminary title and introduction, the model suggests a statistically likely conclusion or abstract body based on the current distribution of astronomical literature.
