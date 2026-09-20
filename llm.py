from dotenv import load_dotenv
import os
import requests

load_dotenv()



OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

# Fixed free model instead of the openrouter/free router.
MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free"
).strip()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# LLM Query
# ============================================================

def query_llm(question: str, context: str) -> str:

    if not OPENROUTER_API_KEY:
        return "❌ OPENROUTER_API_KEY not found in .env"

    if not question.strip():
        return "❌ Please enter a question."

    if not context.strip():
        return "❌ No context was provided."

    prompt = f"""
You are a document question-answering assistant.

Answer the question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer is not present in the context, say:
   "I could not find the answer in the provided context."
4. Give a clear and concise answer.
5. Answer the question directly.

Context:
{context}

Question:
{question}

Answer:
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "DocSpeak"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=90
        )

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "error",
                    {}
                ).get(
                    "message",
                    response.text
                )
            except Exception:
                error_message = response.text

            return (
                f"❌ OpenRouter API error "
                f"({response.status_code}): {error_message}"
            )

        result = response.json()

        if "error" in result:
            return f"❌ OpenRouter error: {result['error']}"

        choices = result.get("choices", [])

        if not choices:
            return (
                "❌ OpenRouter returned no choices.\n\n"
                f"Response: {result}"
            )

        message = choices[0].get("message", {})
        answer = message.get("content")

        if answer is None:
            return (
                "❌ The model returned no text answer.\n\n"
                f"Response: {result}"
            )

        answer = str(answer).strip()

        if not answer:
            return "❌ The model returned an empty answer."

        return answer

    except requests.exceptions.Timeout:
        return "❌ OpenRouter request timed out."

    except requests.exceptions.RequestException as e:
        return f"❌ OpenRouter connection error: {e}"

    except Exception as e:
        return f"❌ Error: {e}"


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    question = input("❓ Enter your question: ")

    context = """
India, a land of vibrant contrasts and rich heritage, is a nation
renowned for its diverse cultures, ancient traditions, and captivating
history. From the towering Himalayas in the north to the serene
backwaters of Kerala in the south, India's geography is as diverse as
its people. The country is home to multiple languages, religions, and
festivals, all coexisting in a unique blend of unity and diversity.

India's history stretches back thousands of years, with its civilization
emerging as one of the world's oldest. From the Indus Valley civilization
to the Mughal Empire and British rule, India has witnessed the rise and
fall of many empires.

Culturally, India is a kaleidoscope of traditions, customs, and artistic
expressions. Classical dance forms include Bharatanatyam and Kathak.
Major festivals include Diwali, Holi, and Eid.

India has also made significant contributions to science, mathematics,
and literature, including the concept of zero and the decimal system.
In modern times, India has emerged as a global leader in information
technology, with a thriving software industry and innovative startups.

India's unity in diversity is one of its defining characteristics.
Despite many languages, religions, and customs, Indians share a common
identity as citizens of one nation.
"""

    print("\n🤖 Model:", MODEL)
    print("\n📌 Answer:")
    print(query_llm(question, context))
