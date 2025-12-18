import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

# Import models
from models import ProcessRequest, ProcessResponse, BrainItem, SocialRequest, BreakdownRequest, BreakdownResponse, AskRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/build/web"))
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def read_root():
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Brain Dump Backend is Live (Web App build not found)"}



# --- LangChain & Groq Setup ---

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set. AI features will not work.")

# Fallback Models List (Priority Order)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

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

# Global Prompt Objects
dump_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{text}")
])
try:
    dump_parser = JsonOutputParser(pydantic_object=ProcessResponse)
except Exception as e:
    print(f"Validation Error init parser: {e}")
    dump_parser = None

# --- Helper Function for Robustness ---

# --- Helper Function for Robustness ---

def extract_json(content: str):
    """Attempt to extract JSON from a string (even with markdown blocks)."""
    try:
        content = content.strip()
        # Remove Markdown
        if "```" in content:
            start = content.find('[') if '[' in content else content.find('{')
            end = content.rfind(']') if ']' in content else content.rfind('}')
            if start != -1 and end != -1:
                content = content[start:end+1]
        
        # Simple bounds check
        if not (content.startswith('{') or content.startswith('[')):
            # Try finding first { or [
            start_obj = content.find('{')
            start_arr = content.find('[')
            
            start = -1
            if start_obj != -1 and start_arr != -1:
                start = min(start_obj, start_arr)
            elif start_obj != -1:
                start = start_obj
            elif start_arr != -1:
                start = start_arr
                
            if start != -1:
                content = content[start:]
                end_obj = content.rfind('}')
                end_arr = content.rfind(']')
                end = max(end_obj, end_arr)
                if end != -1:
                    content = content[:end+1]

        return json.loads(content)
    except Exception as e:
        print(f"JSON Parsing Error: {e} \nContent: {content[:100]}...")
        return None

async def run_groq(input_data=None, prompt_obj=None, parser_obj=None, raw_prompt_str=None):
    """
    Executes Groq LLM with automatic model switching on rate limit (403/429).
    Returns the raw AIMessage content or parsed dict depending on args.
    """
    if not GROQ_API_KEY:
         raise HTTPException(status_code=503, detail="Groq API Key missing")

    for model_name in GROQ_MODELS:
        try:
            # Init LLM
            llm = ChatGroq(model_name=model_name, temperature=0, groq_api_key=GROQ_API_KEY)
            
            response = None
            if raw_prompt_str:
                response = llm.invoke(raw_prompt_str)
            elif prompt_obj:
                if parser_obj:
                    # Try to use parser if available
                    chain = prompt_obj | llm | parser_obj
                    return chain.invoke(input_data)
                else:
                    # No parser, just get message
                    chain = prompt_obj | llm
                    response = chain.invoke(input_data)
            
            # Extract content from AIMessage
            if hasattr(response, 'content'):
                return response.content
            return response

        except Exception as e:
            err = str(e).lower()
            # Catch 403 (Forbidden/Limit) and 429 (Too Many Requests)
            if "403" in err or "rate limit" in err or "429" in err:
                print(f"⚠️ Model {model_name} rate limited. Switching to next...")
                continue # Try next
            else:
                print(f"❌ Model {model_name} error (non-rate-limit): {e}")
                raise e # Stop for non-rate-limit errors
    
    raise HTTPException(status_code=503, detail="AI Service Busy (Rate Limit). Please try again in a moment.")

# --- Endpoints ---

@app.post("/process-dump", response_model=ProcessResponse)
async def process_dump(request: ProcessRequest):
    try:
        # Pass None as parser since verify failed earlier
        result_text = await run_groq(
            input_data={"text": request.text}, 
            prompt_obj=dump_prompt, 
            parser_obj=None 
        )
        
        data = extract_json(result_text)
        if not data:
             # Fallback
             return ProcessResponse(items=[BrainItem(type="note", content=request.text, summary="Note")])
             
        return ProcessResponse(**data)
    except Exception as e:
        print(f"Error processing dump: {e}")
        # Return fallback instead of 500
        return ProcessResponse(items=[BrainItem(type="note", content=request.text, summary="Error processing")])

@app.post("/social-draft")
async def draft_social_post(request: SocialRequest):
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
        response_text = await run_groq(raw_prompt_str=prompt)
        return {"draft": response_text.strip().strip('"')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/break-down", response_model=BreakdownResponse)
async def break_down_task(request: BreakdownRequest):
    prompt = f"""
    You are an expert productivity coach using the "GTD" method. 
    Break down the following complex task into 3-5 smaller, concrete, actionable steps.
    
    Task: "{request.task_content}"
    
    Output specific actions starting with verbs. 
    Example for "Plan Wedding": ["Research venues", "Draft guest list", "Set budget", "Contact caterers"]
    
    Output ONLY a valid JSON list of strings. No extra text.
    """
    try:
        response_text = await run_groq(raw_prompt_str=prompt)
        content = response_text.strip()
        data = extract_json(content)
        
        if data and isinstance(data, list):
             return BreakdownResponse(subtasks=data)
        
        # Fallback if no JSON
        lines = content.split('\n')
        params = [line.strip().strip('- "') for line in lines if line.strip()]
        return BreakdownResponse(subtasks=params)

    except Exception as e:
        print(f"Error breaking down: {e}")
        return BreakdownResponse(subtasks=["Identify next step", "Execute task", "Verify result"])

@app.post("/ask")
async def ask_brain(request: AskRequest):
    # Construct context
    context_parts = []
    for item in request.context_items:
        item_str = f"• {item.summary or item.content}"
        if item.tags:
            item_str += f" (Tags: {', '.join(item.tags)})"
        if item.due_date:
            item_str += f" [Due: {item.due_date}]"
        context_parts.append(item_str)
    
    context_str = "\n".join(context_parts) if context_parts else "No notes available."
    
    history_str = ""
    if request.history:
        history_parts = []
        for msg in request.history[-10:]:
             role = "User" if msg.get('role') == 'user' else "Meera"
             history_parts.append(f"{role}: {msg.get('content')}")
        history_str = "\n".join(history_parts)
    
    prompt = f"""You are Meera, a friendly and helpful AI assistant for the "My Dump Space" app. 
You help users organize their thoughts, tasks, and notes.

The user has the following items in their notes:
{context_str}

Conversation History:
{history_str}

User's New Question: "{request.query}"

Instructions:
- Answer the question directly and helpfully
- Be conversational and friendly
- Use emojis occasionally to be friendly 😊
- Keep responses concise but helpful (2-4 sentences)
- USE THE CONVERSATION HISTORY to understand context.

Your response:"""
    
    try:
        response_text = await run_groq(raw_prompt_str=prompt)
        return {"answer": response_text.strip()}
    except Exception as e:
        print(f"Error asking brain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}

if os.path.isdir("web_build"):
    app.mount("/", StaticFiles(directory="web_build", html=True), name="static")

@app.exception_handler(404)
async def not_found(request, exc):
    return FileResponse('web_build/index.html')

@app.get("/{full_path:path}")
async def serve_static(full_path: str):
    # Skip API routes (just in case, though order should handle it)
    if full_path.startswith("api/") or full_path in ["process-dump", "ask", "social-draft", "break-down", "health"]:
        raise HTTPException(status_code=404, detail="Not Found")
        
    file_path = os.path.join(frontend_path, full_path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
         return FileResponse(index_path)
    return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
