import os
import json
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


# ------------------------- Configuration -------------------------
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in .env file")


# ------------------------- Database ------------------------------
class InMemoryDatabase:
    def __init__(self):
        self.emails = []
        self.counter = 1

    def store_email(self, raw_email: Dict, cleaned_data: Dict,
                    classification: str, summary: str) -> str:
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
        counts = {}
        for record in self.emails:
            cat = record["classification"]
            counts[cat] = counts.get(cat, 0) + 1
        return {"total_emails": len(self.emails), "categories": counts}


# ------------------------- Core Processor ------------------------
class EmailClassifier:
    def __init__(self, api_key: str, model: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def classify_email(self, cleaned_email: Dict) -> Dict:
        from prompts import CLASSIFICATION_PROMPT  # Import prompt from separate file

        prompt = CLASSIFICATION_PROMPT.format(
            sender=cleaned_email['sender'],
            subject=cleaned_email['subject'],
            content=cleaned_email['content']
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)
            return {
                "summary": result.get("summary", ""),
                "category": result.get("category", "Unclassified")
            }
        except json.JSONDecodeError:
            return {"summary": "", "category": "Unclassified (Invalid JSON)"}


class ProcessingService:
    def __init__(self):
        self.db = InMemoryDatabase()
        self.classifier = EmailClassifier(Config.OPENAI_API_KEY, Config.OPENAI_MODEL)

    async def process_email(self, sender: str, subject: str, content: str) -> Dict:
        # Clean email data
        cleaned = {
            "sender": sender.strip() or "Unknown",
            "subject": subject.strip() or "No Subject",
            "content": content.strip() or "No Content"
        }

        # Classify and summarize
        result = await self.classifier.classify_email(cleaned)

        # Store results
        email_id = self.db.store_email(
            raw_email={"sender": sender, "subject": subject, "content": content},
            cleaned_data=cleaned,
            classification=result["category"],
            summary=result["summary"]
        )

        return {
            "id": email_id,
            "summary": result["summary"],
            "category": result["category"],
            "clean_data": cleaned
        }

    def get_statistics(self) -> Dict:
        return self.db.get_stats()


# ------------------------- Usage Example -------------------------
async def main():
    processor = ProcessingService()

    # Example email processing
    result = await processor.process_email(
        sender="HR Department",
        subject="Benefits Enrollment Deadline",
        content="Complete your benefits enrollment by Friday..."
    )

    print("Processing Result:", result)
    print("Statistics:", processor.get_statistics())


if __name__ == "__main__":
    asyncio.run(main())