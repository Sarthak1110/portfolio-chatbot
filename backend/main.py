from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
from pypdf import PdfReader
import os
import json
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

# STEP 1: Set up the connection to the Groq API (free tier).
load_dotenv(Path(__file__).parent / ".env")  # reads secrets (like GROQ_API_KEY) from backend/.env
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("GROQ_API_KEY is missing from backend/.env")

client=Groq(api_key=my_api_key)  # client = our handle for making AI calls
model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


app = FastAPI()

# Allow the portfolio frontend (Vite dev server / deployed site) to call this API
# from the browser. Set CHATBOT_ALLOWED_ORIGINS in .env (comma-separated) in
# production; defaults to "*" for local development.
_origins = os.getenv("CHATBOT_ALLOWED_ORIGINS", "*")
allow_origins = ["*"] if _origins.strip() == "*" else [
    o.strip() for o in _origins.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []

# Shape a parsed resume must have.
class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    total_experience_years: float | None = None

    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []

class ChatRequest(BaseModel):
    question: str


def ask_candidate(question: str, resume: Resume):

    system_prompt = f"""
You are an AI assistant representing a job candidate.

Below is everything you know about the candidate.

{resume.model_dump_json(indent=2)}

Rules:

1. Answer only using this information.

2. Never hallucinate.

3. If information is unavailable,
say

"I don't have enough information to answer that."

4. Be professional.

5. Answer as if HR is interviewing this candidate.
"""

    response = client.chat.completions.create(

        model=model,

        messages=[

            {
                "role":"system",
                "content":system_prompt
            },

            {
                "role":"user",
                "content":question
            }

        ]

    )

    return response.choices[0].message.content

resume_schema = Resume.model_json_schema()
# Turn raw resume text (extracted from a PDF/docx) into a
# structured Resume object, using the AI.
def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"""
    Parse the following resume:

    {resume_text}
    """
    message_system={
        "role" : "system",
        "content" : system_prompt
    }
    message_user={
        "role" : "user",
        "content" : user_prompt
    }
    messages=[message_system, message_user]
    response_format={
        "type": "json_object"
    }
    response=client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content
    if not raw_output:
        raise ValueError("Model returned an empty response")
    data = json.loads(raw_output)
    resume = Resume(**data)
    return resume


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# Resolve the resume relative to this file so the server works no matter which
# directory it's launched from.
RESUME_PATH = Path(__file__).parent / "Sarthak_Resume (11).pdf"

# Parse the resume once and reuse it, instead of re-parsing (two LLM calls) on
# every request.
_resume_cache: Resume | None = None


def get_resume() -> Resume:
    global _resume_cache
    if _resume_cache is None:
        resume_text = read_pdf(RESUME_PATH)
        _resume_cache = parse_resume(resume_text)
    return _resume_cache


@app.get("/health")
def health():
    # Lightweight endpoint for uptime pingers (UptimeRobot/cron-job.org).
    # Makes NO LLM call, so pinging it every few minutes is instant and free.
    return {"status": "ok"}


@app.get("/")
def home():
    resume = get_resume()
    return {
        "message": "hiremeAI resume parsed",
        "name": resume.name,
    }

@app.post("/chat")
def chat(request: ChatRequest):
    resume = get_resume()
    answer = ask_candidate(request.question, resume)
    return {
        "answer": answer
    }


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Streams the AI response token-by-token using Server-Sent Events (SSE).
    The frontend can read the stream line by line and update the UI in real time.
    """
    resume = get_resume()

    system_prompt = f"""
You are an AI assistant representing a job candidate.

Below is everything you know about the candidate.

{resume.model_dump_json(indent=2)}

Rules:

1. Answer only using this information.

2. Never hallucinate.

3. If information is unavailable,
say

"I don't have enough information to answer that."

4. Be professional.

5. Answer as if HR is interviewing this candidate.
"""

    def event_stream():
        """Generator function that yields Server-Sent Event formatted data."""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.question},
                ],
                stream=True,
            )

            for chunk in stream:
                # The final usage-only chunk can have an empty choices list;
                # skip it so we don't IndexError and clobber the streamed answer.
                if not chunk.choices:
                    continue
                # Each chunk contains one or more tokens
                delta = chunk.choices[0].delta
                if delta.content:
                    # SSE format: "data: <json>\n\n"
                    yield f"data: {json.dumps({'token': delta.content})}\n\n"

            # Signal end of stream
            yield "data: [DONE]\n\n"

        except Exception as e:
            # Send error in SSE format
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )
