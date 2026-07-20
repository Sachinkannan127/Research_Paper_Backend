import os
import sys

# Ensure backend_llm workspace root and app path are in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.routes.stream import _stream_answer

try:
    print("Starting generator test...")
    generator = _stream_answer(model_name="fast", question="Hello")
    for chunk in generator:
        print(f"YIELDED: {repr(chunk)}")
    print("Generator completed successfully.")
except Exception as e:
    import traceback
    print("Generator raised an exception:")
    traceback.print_exc()
