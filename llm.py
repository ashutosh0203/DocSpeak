from dotenv import load_dotenv
import os
load_dotenv()

# --- HF Model Config ---
HF_TOKEN = os.getenv("HF_TOKEN") 
from huggingface_hub import InferenceClient
MODEL = "deepset/roberta-base-squad2"

client = InferenceClient(model=MODEL, token=HF_TOKEN)

def query_llm(question: str, context: str) -> str:
    try:
        result = client.question_answering(question=question, context=context)
        return result.get("answer", "❌ No answer found.")
    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == "__main__":
    question = input("❓ Enter your question: ")

    context = """
    India, a land of vibrant contrasts and rich heritage, is a nation renowned for its diverse cultures, ancient traditions, and captivating history. From the towering Himalayas in the north to the serene backwaters of Kerala in the south, India's geography is as diverse as its people. The country is home to multiple languages, religions, and festivals, all coexisting in a unique blend of unity and diversity. This harmonious coexistence of different traditions and modernity is a hallmark of India, drawing visitors from across the globe. 
India's history stretches back thousands of years, with its civilization emerging as one of the world's oldest. From the Indus Valley civilization to the Mughal Empire and British rule, India has witnessed the rise and fall of many empires, each leaving its indelible mark on the nation's cultural landscape. This rich historical tapestry is reflected in the numerous forts, temples, and monuments that dot the country, each narrating tales of bygone eras. 
Culturally, India is a kaleidoscope of traditions, customs, and artistic expressions. From classical dance forms like Bharatanatyam and Kathak to vibrant festivals like Diwali, Holi, and Eid, India's cultural diversity is a spectacle to behold. The concept of "Atithi Devo Bhava," which translates to "the guest is equivalent to God," is deeply ingrained in Indian culture, reflecting the warmth and hospitality extended to visitors. 
India's contribution to the world extends beyond its cultural heritage. It has been a cradle of spirituality and philosophy, with religions like Hinduism, Buddhism, Jainism, and Sikhism originating here. The country has also made significant contributions to science, mathematics, and literature, with inventions like the concept of zero and the decimal system. In modern times, India has emerged as a global leader in information technology, with its thriving software industry and innovative startups. 
India's unity in diversity is perhaps its most defining characteristic. Despite the multitude of languages, religions, and customs, Indians share a common identity as citizens of one nation. This spirit of unity is celebrated through various festivals and cultural events, fostering a sense of belonging and national pride. 
Looking ahead, India is poised to play a significant role on the global stage. With a young and dynamic population, a thriving economy, and a rich cultural heritage, India has the potential to become a global superpower in the 21st century. However, challenges like poverty, inequality, and environmental sustainability need to be addressed to ensure that India's growth benefits all its citizens. 
In conclusion, India is a land of captivating beauty, rich history, and vibrant culture. Its unity in diversity, its ancient traditions, and its modern aspirations make it a unique and fascinating nation, one that continues to inspire and captivate the world. 
    """

    print("\n📌 Answer:", query_llm(question, context))
