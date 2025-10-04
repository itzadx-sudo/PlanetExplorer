import http.server
import socketserver
import webbrowser
import os
import sys


def serve_locally(html_file="planetexplorer_final.html", port=8000):
    """Start a local HTTP server and open the HTML file in browser"""
    
    # Check if HTML file exists
    if not os.path.exists(html_file):
        print(f"⚠️  {html_file} not found. Generating it first...")
        return False
    
    # Get the directory containing the HTML file
    html_dir = os.path.dirname(os.path.abspath(html_file))
    html_filename = os.path.basename(html_file)
    
    # Change to the directory containing the HTML file
    if html_dir:
        os.chdir(html_dir)
    
    # Create server
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}/{html_filename}"
            print("=" * 70)
            print(f"🌐 Local server started at: {url}")
            print(f"📂 Serving from: {os.getcwd()}")
            print("=" * 70)
            print("\n✨ Opening browser...")
            print("⏹️  Press Ctrl+C to stop the server\n")
            
            # Open browser
            webbrowser.open(url)
            
            # Keep server running
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        sys.exit(0)
    except OSError as e:
        if e.errno == 48 or e.errno == 98:  # Address already in use
            print(f"⚠️  Port {port} is already in use. Try another port.")
            print(f"   Run with a different port number")
        else:
            print(f"❌ Error starting server: {e}")
        return False
    
    return True


def main():
    print("=" * 70)
    print("   NASA Space Apps Challenge 2025")
    print("=" * 70)
    print()


    output_file = "lumina_final.html"

    print("=" * 70)
    
    # Start local server automatically
    print("Starting local server...\n")
    serve_locally(output_file)


if __name__ == "__main__":
    main()
