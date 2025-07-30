#!/usr/bin/env python3
import os
import sys
import uvicorn

print("Python path:", sys.path)
print("Current working directory:", os.getcwd())
print("Files in directory:", os.listdir("."))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on host 0.0.0.0 port {port}")
    print("Environment PORT:", os.environ.get("PORT", "Not set"))
    
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="debug")
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()