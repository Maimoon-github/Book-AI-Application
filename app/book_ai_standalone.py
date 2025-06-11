# -*- coding: utf-8 -*-
"""
Book AI Processor - Final Standalone Version for PyInstaller
This version runs Streamlit programmatically without subprocess calls
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
    """Run the Streamlit app by directly calling the main script programmatically"""
    try:
        print("🚀 Starting BookAI Application...")
        print("📚 Book AI Processor with Teaching Assistant")
        print("=" * 50)
        logging.info("BookAI Application starting")
        
        # Find a free port
        port = find_free_port()
        print(f"🌐 Starting server on port {port}...")
        logging.info(f"Using port {port}")
        
        # Set up environment
        os.environ['STREAMLIT_SERVER_PORT'] = str(port)
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
        
        def start_server():
            try:
                print("📦 Importing required modules...")
                logging.info("Importing required modules")
                
                # Import streamlit and configure it
                import streamlit as st
                from streamlit import config
                
                # Configure Streamlit
                config.set_option('server.port', port)
                config.set_option('server.headless', True)
                config.set_option('browser.gatherUsageStats', False)
                config.set_option('server.address', 'localhost')
                config.set_option('global.developmentMode', False)
                
                print("🔧 Configuring Streamlit server...")
                logging.info("Configuring Streamlit server")
                
                # Instead of using CLI, run the server programmatically
                from streamlit.web.server import Server
                from streamlit.web.server.server import start_listening
                
                print("📥 Loading BookAI application...")
                logging.info("Loading BookAI application")
                
                # Import the book_ai module directly
                import book_ai
                
                # Get the script path for Streamlit
                script_path = book_ai.__file__ if hasattr(book_ai, '__file__') else 'book_ai'
                
                print(f"🎯 Starting Streamlit server with script: {script_path}")
                logging.info(f"Starting Streamlit server with script: {script_path}")
                
                # Create server instance
                server = Server(
                    script_path,
                    command_line='',
                    is_hello=False
                )
                
                # Start the server
                server.start()
                
                print("✅ Streamlit server started successfully")
                logging.info("Streamlit server started successfully")
                
                # Keep the server running
                server._server.serve_forever()
                
            except Exception as e:
                print(f"❌ Error starting Streamlit server: {e}")
                logging.error(f"Error starting Streamlit server: {e}")
                
                # Fallback: try direct import and manual setup
                try:
                    print("🔄 Trying fallback approach...")
                    logging.info("Trying fallback approach")
                    
                    # Simple direct import and execution
                    import book_ai
                    
                    # Use streamlit's run functionality directly
                    import streamlit.bootstrap as bootstrap
                    
                    # Modify sys.argv to simulate command line
                    old_argv = sys.argv.copy()
                    sys.argv = ['streamlit', 'run', '--server.port', str(port)]
                    
                    # Run the app
                    bootstrap.run('book_ai', '', [], {})
                    
                    print("✅ Fallback approach successful")
                    logging.info("Fallback approach successful")
                    
                except Exception as e2:
                    print(f"❌ Fallback also failed: {e2}")
                    logging.error(f"Fallback failed: {e2}")
                    
                    # Final fallback: just import and hope for the best
                    try:
                        print("🔄 Final fallback: direct import...")
                        logging.info("Final fallback: direct import")
                        
                        # Direct import should execute the streamlit code
                        import book_ai
                        
                        print("✅ Direct import completed")
                        logging.info("Direct import completed")
                        
                        # Keep thread alive
                        while True:
                            time.sleep(1)
                            
                    except Exception as e3:
                        print(f"❌ All approaches failed: {e3}")
                        logging.error(f"All approaches failed: {e3}")
                finally:
                    try:
                        sys.argv = old_argv
                    except:
                        pass
        
        # Start server in background thread
        print("🧵 Starting server thread...")
        logging.info("Starting server thread")
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Give server time to start
        print("⏳ Initializing server...")
        for i in range(15):
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
                    print("⚠️  Server thread stopped")
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