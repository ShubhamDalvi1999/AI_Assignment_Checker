"""
Simple script to test that the logging configuration works properly.
This will generate logs at different levels to test all handlers.
"""
import logging
import os
import sys

# First, import the logging configuration from the main app
# This will initialize the logging system
try:
    # Only import the logging-related part of the app
    from app import logger
    
    # Get the current log directory location
    import app
    if hasattr(app, 'logs_dir'):
        print(f"Logs directory is configured as: {app.logs_dir}")
        
        # Check if directory exists and is writable
        if os.path.exists(app.logs_dir):
            print(f"✅ Logs directory exists at: {app.logs_dir}")
            if os.access(app.logs_dir, os.W_OK):
                print(f"✅ Logs directory is writable")
            else:
                print(f"❌ Logs directory is NOT writable")
        else:
            print(f"❌ Logs directory does NOT exist at: {app.logs_dir}")
    else:
        print("⚠️ Could not determine logs directory from app module")

    # Now generate some log entries at different levels
    print("\nGenerating test log entries...")
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    # Generate a log that should go to the API log file
    logger.info("This is an OpenAI API test message that should go to api_calls.log")
    
    # Check what handlers are configured
    root_logger = logging.getLogger()
    print(f"\nLogging handlers configured: {len(root_logger.handlers)}")
    for i, handler in enumerate(root_logger.handlers):
        handler_type = type(handler).__name__
        if hasattr(handler, 'baseFilename'):
            print(f"  {i+1}. {handler_type}: {handler.baseFilename}")
        else:
            print(f"  {i+1}. {handler_type}")
    
    print("\n✅ Test completed successfully. Check the log files to verify entries were created.")
    
except Exception as e:
    print(f"❌ Error during logging test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 