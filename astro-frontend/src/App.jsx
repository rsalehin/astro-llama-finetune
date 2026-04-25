import React, { useState } from 'react';
import { Rocket, Sparkles, Send } from 'lucide-react';

export default function App() {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleGenerate = async () => {
    if (!input) return;
    setLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: input }),
      });
      const data = await response.json();
      setOutput(data.completion);
    } catch (error) {
      setOutput("Error: Backend not responding. Is server.py running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0d17] text-white p-10 font-sans">
      <div className="max-w-3xl mx-auto text-center">
        <Rocket className="mx-auto text-indigo-500 mb-4" size={48} />
        <h1 className="text-4xl font-bold mb-2">Astro-LLaMA v1.0</h1>
        <p className="text-gray-400 mb-8">Fine-tuned Astronomy Abstract Generator</p>
        
        <div className="bg-[#161b22] p-6 rounded-xl border border-[#30363d] text-left">
          <textarea 
            className="w-full h-40 bg-[#0d1117] border border-[#30363d] rounded-lg p-4 text-white focus:outline-none focus:border-indigo-500"
            placeholder="Enter Title and Introduction..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button 
            onClick={handleGenerate}
            disabled={loading}
            className="w-full mt-4 bg-indigo-600 hover:bg-indigo-500 py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all"
          >
            {loading ? "Computing..." : <><Sparkles size={20}/> Generate Abstract</>}
          </button>
        </div>

        {output && (
          <div className="mt-8 p-6 bg-[#161b22] border-l-4 border-indigo-500 text-left">
            <h3 className="text-indigo-400 font-bold flex items-center gap-2 mb-2">
              <Send size={16}/> Completion:
            </h3>
            <p className="text-gray-300 italic">{output}</p>
          </div>
        )}
      </div>
    </div>
  );
}