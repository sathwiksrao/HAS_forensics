"""
Simple HTTP server for the frontend.
Run this to serve the frontend files.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
FRONTEND_DIR = Path(__file__).resolve().parent

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

if __name__ == "__main__":
    os.chdir(str(FRONTEND_DIR))
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving frontend at http://localhost:{PORT}")
        print(f"Backend should be running at http://127.0.0.1:5000")
        httpd.serve_forever()
