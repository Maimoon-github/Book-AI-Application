# -*- coding: utf-8 -*-
"""
Book AI Processor - Standalone Version for PyInstaller
This version can be run as a standalone executable without requiring `streamlit run`
"""

import sys
import os
import threading
import time
import webbrowser
import logging

def find_free_port():
    """Find a free port for the Streamlit server"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def run_streamlit_app():
    """Run the Streamlit app directly by calling book_ai and setting up streamlit"""
    try:
        print("🚀 Starting BookAI Application...")
        print("📚 Book AI Processor with Teaching Assistant")
        print("=" * 50)
        logging.info("BookAI Application starting")
        
        # Find a free port
        port = find_free_port()
        print(f"🌐 Starting server on port {port}...")
        logging.info(f"Using port {port}")
        
        def start_server():
            try:
                print("🔧 Setting up Streamlit environment...")
                logging.info("Setting up Streamlit environment")
                
                # Set up environment for streamlit
                os.environ['STREAMLIT_SERVER_PORT'] = str(port)
                os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
                os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
                
                print("📦 Importing Streamlit...")
                logging.info("Importing Streamlit")
                # Import and configure streamlit
                import streamlit as st
                from streamlit.web import cli as stcli
                
                print("⚙️ Configuring Streamlit CLI...")
                logging.info("Configuring Streamlit CLI")
                # Set up sys.argv to simulate 'streamlit run book_ai.py'
                original_argv = sys.argv.copy()
                
                # Try to find book_ai.py in the current directory
                current_dir = os.path.dirname(os.path.abspath(__file__))
                book_ai_path = os.path.join(current_dir, 'book_ai.py')
                
                print(f"📂 Looking for book_ai.py at: {book_ai_path}")
                logging.info(f"Looking for book_ai.py at: {book_ai_path}")
                
                if os.path.exists(book_ai_path):
                    print("✅ Found book_ai.py, using file path")
                    logging.info("Found book_ai.py as file")
                    script_target = book_ai_path
                else:
                    print("⚠️  book_ai.py not found as file, trying module import")
                    logging.warning("book_ai.py not found as file, trying module import")
                    script_target = 'book_ai'
                
                sys.argv = [
                    'streamlit', 'run', script_target,
                    '--server.port', str(port),
                    '--server.headless', 'true',
                    '--browser.gatherUsageStats', 'false',
                    '--server.address', 'localhost',
                    '--global.developmentMode', 'false'
                ]
                
                print(f"🎯 Starting Streamlit with: {' '.join(sys.argv)}")
                logging.info(f"Starting Streamlit with: {' '.join(sys.argv)}")
                
                # Start streamlit
                stcli.main()
                
            except Exception as e:
                print(f"❌ Error in Streamlit CLI: {e}")
                logging.error(f"Error in Streamlit CLI: {e}")
                print(f"📋 Error details: {type(e).__name__}: {str(e)}")
                
                # If CLI fails, try direct import approach
                try:
                    print("🔄 Trying direct import approach...")
                    logging.info("Trying direct import approach")
                    sys.argv = original_argv  # Reset argv
                    
                    # Simple import of book_ai
                    print("📥 Importing book_ai module...")
                    logging.info("Importing book_ai module")
                    import book_ai
                    print("✅ book_ai module imported successfully")
                    logging.info("book_ai module imported successfully")
                    
                    # The book_ai module execution should start the streamlit app
                    
                except Exception as e2:
                    print(f"❌ Direct import also failed: {e2}")
                    logging.error(f"Direct import failed: {e2}")
                    print(f"📋 Import error details: {type(e2).__name__}: {str(e2)}")
                    print("\n🔍 Debugging Information:")
                    print(f"📂 Current directory: {os.getcwd()}")
                    print(f"📁 Script directory: {os.path.dirname(os.path.abspath(__file__))}")
                    print(f"🐍 Python path: {sys.path[:3]}...")  # Show first 3 paths
                    logging.error(f"Debugging info - Current dir: {os.getcwd()}")
                    logging.error(f"Script dir: {os.path.dirname(os.path.abspath(__file__))}")
            finally:
                try:
                    sys.argv = original_argv
                except:
                    pass
        
        # Start server in background thread
        print("🧵 Starting server thread...")
        logging.info("Starting server thread")
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Give server time to start
        print("⏳ Initializing server...")
        for i in range(15):  # Increased wait time
            print(f"⏰ Waiting... {i+1}/15")
            time.sleep(1)
        
        # Open browser
        url = f"http://localhost:{port}"
        print(f"🎉 Opening BookAI: {url}")
        logging.info(f"Opening browser to: {url}")
        webbrowser.open(url)
        
        print("\n📖 BookAI should be running!")
        print("🔑 Enter your Groq API key in the sidebar to enable AI features")
        print("❌ Close this window to stop the application")
        print("\n🔧 Debug info:")
        print(f"🌐 Server URL: {url}")
        print(f"🧵 Server thread alive: {server_thread.is_alive()}")
        
        if not server_thread.is_alive():
            print("⚠️  Server thread stopped unexpectedly")
            logging.warning("Server thread stopped unexpectedly")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(5)
                if not server_thread.is_alive():
                    print("⚠️  Server thread stopped - trying to restart...")
                    logging.warning("Server thread stopped")
                    break
                else:
                    print(f"✅ Server still running - {url}")
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            logging.info("Application shutting down via KeyboardInterrupt")
            
    except Exception as e:
        print(f"❌ Critical error starting BookAI: {e}")
        logging.error(f"Critical error: {e}")
        print(f"📋 Error details: {type(e).__name__}: {str(e)}")
        input("\nPress Enter to exit...")

def main():
    """Main entry point"""
    # Add debug logging to file
    logging.basicConfig(
        filename='bookai_debug.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    try:
        print("BookAI Starting - Check bookai_debug.log for details")
        logging.info("BookAI application starting")
        run_streamlit_app()
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.error(f"Fatal error in main: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
