import argparse
import http.server
import socketserver
import webbrowser
import os
import sys


def serve_locally(html_file: str = "index.html", port: int = 8000):
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found.")
        return False

    html_dir = os.path.dirname(os.path.abspath(html_file))
    html_filename = os.path.basename(html_file)

    if html_dir:
        os.chdir(html_dir)

    Handler = http.server.SimpleHTTPRequestHandler

    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}/{html_filename}"
            print(f"Local server started at: {url}")
            print(f"Serving from: {os.getcwd()}")
            print("Press Ctrl+C to stop the server")
            webbrowser.open(url)
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
    except OSError as e:
        if e.errno in (48, 98):
            print(f"Port {port} is already in use. Try a different port with --port.")
        else:
            print(f"Error starting server: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the PlanetExplorer frontend")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    args = parser.parse_args()
    serve_locally("index.html", port=args.port)
