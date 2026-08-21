import json
import urllib.request
import urllib.error
from app.core.config import settings

def search_exa(query: str, num_results: int = 5) -> dict:
    """
    Perform a search using the Exa API and return raw results.
    """
    api_key = settings.EXA_API_KEY
    if not api_key or api_key == "YOUR_API_KEY":
        return {"error": "Exa API key not configured or is set to placeholder."}

    url = "https://api.exa.ai/search"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "type": "auto",
        "numResults": num_results,
        "contents": {
            "highlights": True
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_data = response.read().decode("utf-8")
            return json.loads(res_data)
    except urllib.error.HTTPError as e:
        try:
            error_body = e.read().decode("utf-8")
            return {"error": f"HTTP Error {e.code}: {e.reason}", "details": error_body}
        except Exception:
            return {"error": f"HTTP Error {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}
