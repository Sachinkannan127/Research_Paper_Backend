import json
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG_FILE = os.path.join("app", "core", "rag_config.json")

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HP_API_KEY = os.getenv("HP_API_KEY")
    
    @staticmethod
    def get_default_config():
        # Import system prompt as default config value
        from app.prompts.system_prompt import SYSTEM_PROMPT
        return {
            "active_pdf_name": "Research_paper.pdf",
            "active_pdf_path": os.path.join("app", "uploads", "Research_paper.pdf"),
            "system_prompt": SYSTEM_PROMPT,
            "welcome_message": "Ask me anything about the uploaded research paper!"
        }

    def load_rag_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Merge with defaults in case of missing keys
                    defaults = self.get_default_config()
                    for k, v in defaults.items():
                        if k not in data:
                            data[k] = v
                    return data
            except Exception as e:
                print(f"Error loading RAG config: {e}")
        return self.get_default_config()

    def save_rag_config(self, config_dict):
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving RAG config: {e}")

settings = Settings()
