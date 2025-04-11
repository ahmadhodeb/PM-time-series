import http.server
import socketserver
import os
import subprocess
import psutil

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        # Handle POST requests
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


def start_server():
    # Change directory to the project folder
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Free the port before starting the server
    PORT = 8000
    free_port(PORT)

    # Start the HTTP server
    handler = CustomHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving HTTP on port {PORT}...")
        httpd.serve_forever()

if __name__ == "__main__":
    try:
        # Start the backend server
        subprocess.Popen(["python", "api_inference.py"])

        # Start the HTTP server
        start_server()
    except KeyboardInterrupt:
        print("Shutting down the server...")