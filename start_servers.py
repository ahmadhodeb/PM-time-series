import http.server
import socketserver
import os
import subprocess
import psutil
import sys
import time

print('[DEBUG] start_servers.py: Starting server...')

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"POST request received")

def free_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for conn in proc.connections(kind='inet'):
                if conn.laddr.port == port:
                    print(f"Terminating process {proc.info['name']} (PID: {proc.info['pid']}) using port {port}.")
                    proc.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

def start_static_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    PORT = 8000
    free_port(PORT)
    handler = CustomHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving static files on port {PORT}...")
        httpd.serve_forever()

def start_fastapi_server():
    # Start FastAPI backend on port 8001 using uvicorn
    print("Starting FastAPI backend on port 8001...")
    subprocess.Popen([sys.executable, "-m", "uvicorn", "api_inference:app", "--host", "127.0.0.1", "--port", "8001", "--reload"])
    # Give backend a moment to start
    time.sleep(2)

if __name__ == "__main__":
    print('[DEBUG] start_servers.py: Server script loaded.')
    try:
        start_fastapi_server()
        start_static_server()
    except KeyboardInterrupt:
        print("Shutting down the server...")