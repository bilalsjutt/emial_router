import os
import json
from typing import Dict, Any
from fastapi import FastAPI, APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
# import openai
from openai import AsyncOpenAI
# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in .env file")

    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # Use an in-memory database if set
    USE_IN_MEMORY_DB = os.getenv("USE_IN_MEMORY_DB", "true").lower() == "true"

# -----------------------------------------------------------------------------
# In-Memory Database for Dev
# -----------------------------------------------------------------------------
class InMemoryDatabase:
    def __init__(self):
        self.emails = []
        self.counter = 1

    async def store_email(
        self,
        raw_email: Dict[str, Any],
        cleaned_data: Dict[str, str],
        classification: str,
        summary: str
    ) -> str:
        record = {
            "id": str(self.counter),
            "raw_email": raw_email,
            "cleaned_data": cleaned_data,
            "classification": classification,
            "summary": summary
        }
        self.emails.append(record)
        self.counter += 1
        return record["id"]

    def get_stats(self) -> Dict[str, Any]:
        category_counts = {}
        for record in self.emails:
            cat = record["classification"].lower()
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_emails": len(self.emails),
            "categories": category_counts
        }

# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
class EmailMessage(BaseModel):
    sender: str
    subject: str
    content: str

class EmailResponse(BaseModel):
    id: str
    summary: str
    category: str
    clean_data: Dict[str, str]

# -----------------------------------------------------------------------------
# Email Processor (Simulating MCP)
# -----------------------------------------------------------------------------
class EmailProcessor:
    """
    Cleans and validates incoming email data.
    In production, replace this with your actual MCP service calls.
    """
    def clean_email(self, email: EmailMessage) -> Dict[str, str]:
        return {
            "sender": email.sender.strip() if email.sender else "Unknown",
            "subject": email.subject.strip() if email.subject else "No Subject",
            "content": email.content.strip() if email.content else "No Content"
        }

# -----------------------------------------------------------------------------
# Email Classifier using OpenAI (Latest Async ChatCompletion)
# -----------------------------------------------------------------------------
# ... (keep all other imports the same)



# ... (keep Config and InMemoryDatabase the same)

# -----------------------------------------------------------------------------
# Email Classifier using OpenAI (Updated for v1.0+ API)
# -----------------------------------------------------------------------------
class EmailClassifier:
    def __init__(self, api_key: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key)  # New client initialization
        self.model = model

    async def classify_and_summarize(self, cleaned_email: Dict[str, str]) -> Dict[str, str]:
        prompt = (
            "You are an expert email assistant. Analyze the email below.\n"
            "1) If the email is long (over 100 words), summarize it in 2-3 lines.\n"
            "2) Classify the email into the following categories:\n"
            "   - Critical (urgent/time-sensitive)\n"
            "   - Proposal (budget/project proposals)\n"
            "   - HR (employee matters/benefits)\n"
            "   - Finance (budgets/expenses)\n"
            "   - Benefits (enrollments/updates)\n"
            "   - Marketing (campaigns/promotions)\n"
            "   - Professional (work-related)\n"
            "   - Personal (non-work related)\n"
            "   - Actions\n"
            "   - Storage\n"
            "   - Trash\n"
            "   - Labels (for any additional custom categories provided by the user, such as friends, family, etc.)\n"
            "3) For long emails, store the generated summary in a daily summary folder.\n\n"
            f"Sender: {cleaned_email['sender']}\n"
            f"Subject: {cleaned_email['subject']}\n"
            f"Content: {cleaned_email['content']}\n\n"
            "Return JSON: {\"summary\": \"...\", \"category\": \"...\"}"
        )

        messages = [
            {"role": "system", "content": "You are an expert email assistant."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Updated API call for v1.0+
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}  # Force JSON response
            )

            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)

            return {
                "summary": result.get("summary", ""),
                "category": result.get("category", "Unclassified")
            }

        except json.JSONDecodeError:
            return {"summary": "", "category": "Unclassified (Invalid JSON)"}
        except Exception as e:
            print(f"OpenAI API Error: {str(e)}")
            raise HTTPException(
                status_code=502,
                detail=f"AI service error: {str(e)}"
            )


# ... (rest of the code remains the same)
# -----------------------------------------------------------------------------
# Email Router Service
# -----------------------------------------------------------------------------
class EmailRouterService:
    def __init__(self, openai_api_key: str, database, model: str):
        print("EmailRouterService _init_ called with:", openai_api_key, model)
        self.router = APIRouter()
        self.email_processor = EmailProcessor()
        self.email_classifier = EmailClassifier(api_key=openai_api_key, model=model)
        self.database = database
        self._setup_routes()

    def _setup_routes(self):
        @self.router.post("/classify", response_model=EmailResponse)
        async def classify_email(email: EmailMessage = Body(...)):
            try:
                # Clean email data
                cleaned_email = self.email_processor.clean_email(email)
                # Summarize & classify
                result = await self.email_classifier.classify_and_summarize(cleaned_email)
                # Store the email data
                email_id = await self.database.store_email(
                    raw_email=email.dict(),
                    cleaned_data=cleaned_email,
                    classification=result["category"],
                    summary=result["summary"]
                )
                return {
                    "id": email_id,
                    "summary": result["summary"],
                    "category": result["category"],
                    "clean_data": cleaned_email
                }
            except Exception as e:
                print(f"Error processing email: {str(e)}")
                if isinstance(e, HTTPException):
                    raise e
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing email: {str(e)}"
                )

        @self.router.get("/stats")
        async def get_stats():
            return self.database.get_stats()

        @self.router.get("/gmail_status")
        async def gmail_status():
            return {
                "connected": False,
                "message": "Not Connected"
            }

# -----------------------------------------------------------------------------
# FastAPI Application Factory
# -----------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="Mail Summarizer & Classifier",
        description="Classifies emails into categories and provides stats. Also simulates Gmail status.",
        version="1.0.0"
    )

    # Choose in-memory or real DB
    if Config.USE_IN_MEMORY_DB:
        database = InMemoryDatabase()
    else:
        # In production, connect to a real database
        database = InMemoryDatabase()

    email_router_service = EmailRouterService(
        openai_api_key=Config.OPENAI_API_KEY,
        database=database,
        model=Config.OPENAI_MODEL
    )

    # Register routes under /email
    app.include_router(email_router_service.router, prefix="/email", tags=["Email Service"])
    return app

app = create_app()

# -----------------------------------------------------------------------------
# Run the server:
#   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# -----------------------------------------------------------------------------