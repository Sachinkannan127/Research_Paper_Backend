import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # Bind to 0.0.0.0 to listen on all network interfaces (localhost, Wi-Fi IP, and Render deployment container)
    host = os.getenv("HOST", "0.0.0.0")
    reload = False if "PORT" in os.environ else True
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
    