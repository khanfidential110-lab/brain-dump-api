import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

# Import models
# Note: If running this file directly, ensure 'models.py' is in the Python path.
from models import ProcessRequest, ProcessResponse, BrainItem, SocialRequest, BreakdownRequest, BreakdownResponse, AskRequest

app = FastAPI()

# Mount API routes heavily first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SERVE FLUTTER WEB APP
# Must come after API routes or be mounted specifically? 
# Best practice: Mount static on / and have API on /api or similar.
# But for this MVP root dump, we can just serve static fallback.

# We will mount static files for assets
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/build/web"))
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def read_root():
    # Return index.html if exists, else API msg
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Brain Dump Backend is Live (Web App build not found)"}

# Serve other static files (js, css, etc) from root if not matching API
# This acts as a catch-all for the Single Page App (SPA)
@app.get("/{full_path:path}")
async def serve_static(full_path: str):
    # Check if file exists in build/web
    file_path = os.path.join(frontend_path, full_path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    # SPA Fallback for routes managed by Flutter
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
         return FileResponse(index_path)
    return {"error": "File not found"}

# --- LangChain Setup ---

# SWITCHED TO GROQ (Cloud API - Fast & Reliable)
from langchain_groq import ChatGroq
import os

# Get Groq API Key from environment (set in Render dashboard)
# DO NOT hardcode API keys - set GROQ_API_KEY environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set. AI features will not work.")

# Initialize Groq with Llama 3.3 70B (fast & powerful)
try:
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    print("Connected to Groq Cloud API (Llama 3.3 70B)")
except Exception as e:
    print(f"Warning: Could not connect to Groq: {e}")
    llm = None

# System Prompt
# We strictly enforce JSON output using the schema from our Pydantic models (implicitly)
# but explicitly in the prompt for the LLM to follow.
SYSTEM_PROMPT = """
You are an intelligent assistant for the "Brain Dump" app, designed to help people with ADHD organize their thoughts.
Your goal is to parse unstructured thoughts into structured lists with helpful metadata.

Categorize the input text into one of the following types:
- "task": A to-do item.
- "event": A calendar event (look for time/date mentions).
- "shopping": An item to buy.
- "note": General information, ideas, or things that don't fit other categories.

You must output a JSON object with a single key "items" which is a list of objects.
Each object must have:
- "type": One of ["task", "event", "shopping", "note"]
- "content": The original text of the item.
- "date": A string representation of the date/time if present, otherwise null.
- "summary": A concise, 3-5 word summary of the item for quick scanning.
- "tags": A list of short tags (e.g. "Work", "Urgent", "Health", "Idea").
- "priority": One of ["High", "Medium", "Low"] based on urgency or importance.
- "sentiment": One of ["Positive", "Neutral", "Negative"].
- "emoji": A single emoji representing the "vibe" (e.g. 🔥, 💙, ✅, 📅, 🛒, 💡).
- "due_date": If a specific time is mentioned (e.g. "tomorrow at 5pm"), extract it as an ISO 8601 string (YYYY-MM-DDTHH:MM:SS), assuming "now" is the current time. If no time, return null.

Example Input: "Buy milk and meeting with Sarah tomorrow at 2pm"
Example Output:
{{
  "items": [
    {{ 
      "type": "shopping", 
      "content": "Buy milk", 
      "date": null,
      "summary": "Milk", 
      "tags": ["Groceries", "Home"], 
      "priority": "Medium",
      "sentiment": "Neutral",
      "emoji": "🛒",
      "due_date": null
    }},
    {{ 
      "type": "event", 
      "content": "Meeting with Sarah", 
      "date": "tomorrow at 2pm",
      "summary": "Sarah Meeting", 
      "tags": ["Social", "Schedule"], 
      "priority": "High",
      "sentiment": "Positive",
      "emoji": "📅",
      "due_date": "2024-12-17T14:00:00"
    }}
  ]
}}

Only output valid JSON. No preambles.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{text}")
])

# Using JsonOutputParser to help with parsing (though we'll do manual fallback too for safety)
parser = JsonOutputParser(pydantic_object=ProcessResponse)

# Initialize chain only if LLM is available
if llm:
    chain = prompt | llm | parser
else:
    chain = None

@app.post("/process-dump", response_model=ProcessResponse)
async def process_dump(request: ProcessRequest):
    if not llm:
        raise HTTPException(status_code=500, detail="LLM not initialized. Please ensure Ollama is running.")
    
    try:
        # Invoke the chain
        if not chain:
             raise Exception("LLM Chain not initialized")
        result = chain.invoke({"text": request.text})
        
        # result should be a dict matching ProcessResponse (thanks to JsonOutputParser)
        # We validate it through Pydantic to be sure
        return ProcessResponse(**result)

    except Exception as e:
        print(f"Error processing dump: {e}")
        # In case of parsing error or hallucination that breaks JSON
        raise HTTPException(status_code=500, detail=f"AI Processing Failed: {str(e)}")

@app.post("/social-draft")
async def draft_social_post(request: SocialRequest):
    if not llm:
        raise HTTPException(status_code=503, detail="Ollama not available")
    
    prompt = f"""
    You are a social media expert. Rewrite the following thought into a viral, engaging post for {request.platform.upper()}.
    
    Thought: "{request.content}"
    
    Rules:
    - If Twitter/X: Under 280 chars, use 2-3 relevant hashtags, punchy tone.
    - If Reddit: Conversational, ask a question to start discussion.
    - If LinkedIn: Professional but authentic, use structured formatting (bullet points).
    - If Facebook/Instagram: Casual, emoji-rich.
    
    Output ONLY the post text. No "Here is the post" preamble.
    """
    
    try:
        response = llm.invoke(prompt)
        return {"draft": response.content.strip().strip('"')} # Remove quotes if AI adds them
    except Exception as e:
        print(f"Error drafting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/break-down", response_model=BreakdownResponse)
async def break_down_task(request: BreakdownRequest):
    if not llm:
        raise HTTPException(status_code=503, detail="Ollama not available")
    
    prompt = f"""
    You are an expert productivity coach using the "GTD" method. 
    Break down the following complex task into 3-5 smaller, concrete, actionable steps.
    
    Task: "{request.task_content}"
    
    Output specific actions starting with verbs. 
    Example for "Plan Wedding": ["Research venues", "Draft guest list", "Set budget", "Contact caterers"]
    
    Output ONLY a valid JSON list of strings. No extra text.
    """
    
    try:
        response = llm.invoke(prompt)
        # Clean response to ensure only the list part is parsed
        content = response.content.strip()
        
        # Remove markdown code blocks if present
        if "```" in content:
            # Find the first [ and the last ]
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1:
                content = content[start:end+1]
            else:
                # Fallback: try to just lines that look like items
                lines = content.split('\n')
                params = [line.strip().strip('- "') for line in lines if line.strip()]
                return BreakdownResponse(subtasks=params)
            
        subtasks = json.loads(content)
        return BreakdownResponse(subtasks=subtasks)
    except Exception as e:
        print(f"Error breaking down: {e}")
        # Fallback if JSON fails - return a generic split
        fallback = request.task_content.split(' and ')
        if len(fallback) > 1:
             return BreakdownResponse(subtasks=fallback)
        return BreakdownResponse(subtasks=["Identify next step", "Execute task", "Verify result"])

@app.post("/ask")
async def ask_brain(request: AskRequest):
    if not llm:
        raise HTTPException(status_code=503, detail="Ollama not available")
    
    # Construct context from brain items
    context_str = "\\n".join([f"- {item.type.upper()}: {item.content} (Status: {item.status})" for item in request.context_items])
    
    prompt = f"""
    You are the user's "Second Brain". You have access to their notes and tasks.
    Answer the following question based ONLY on the context provided below.
    
    Context:
    {context_str}
    
    Question: "{request.query}"
    
    Answer clearly and concisely. If the answer is not in the context, say "I don't see that in your notes."
    """
    
    try:
        response = llm.invoke(prompt)
        # Handle potential response object structure (content string vs object)
        answer_text = response.content if hasattr(response, 'content') else str(response)
        return {"answer": answer_text.strip()}
    except Exception as e:
        print(f"Error asking brain: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {"status": "ok"}

# 4. Serve Flutter Web (Static Files)
# Must be AFTER API routes so API takes precedence
if os.path.isdir("web_build"):
    app.mount("/", StaticFiles(directory="web_build", html=True), name="static")

@app.exception_handler(404)
async def not_found(request, exc):
    # Fallback to index.html for Flutter routing (if needed)
    return FileResponse('web_build/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
