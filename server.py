import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    # Bind to 0.0.0.0 in deployment (Render) to allow external routing, fallback to localhost for dev
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    reload = False if "PORT" in os.environ else True
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
    