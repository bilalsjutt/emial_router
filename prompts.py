CLASSIFICATION_PROMPT = """You are an expert email assistant. Analyze the email below.
1) If the email is long (over 100 words), summarize it in 2-3 lines.
2) Classify the email into the following categories:
   - Critical (urgent/time-sensitive)
   - Proposal (budget/project proposals)
   - HR (employee matters/benefits)
   - Finance (budgets/expenses)
   - Benefits (enrollments/updates)
   - Marketing (campaigns/promotions)
   - Professional (work-related)
   - Personal (non-work related)
   - Actions
   - Storage
   - Trash
   - Labels
3) For long emails, store the generated summary in a daily summary folder.

Sender: {sender}
Subject: {subject}
Content: {content}

Return JSON: {{"summary": "...", "category": "..."}}"""