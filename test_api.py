import google.generativeai as genai
import logging
import sys
import traceback

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Your API key
GOOGLE_API_KEY = 'AIzaSyBKJbHSEcB2pSBk9PkrqK50bMBcY3AMwFw'

def test_api_connection():
    try:
        print("\n=== Starting API Test ===")
        print(f"Using API Key: {GOOGLE_API_KEY[:8]}...{GOOGLE_API_KEY[-4:]}")
        
        # Configure the API
        print("\n1. Configuring API...")
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("API configured successfully")
        
        # Get available models
        print("\n2. Fetching available models...")
        models = genai.list_models()
        if not models:
            logger.error("No models available")
            print("❌ API Test Failed: No models available")
            return False
            
        # Print available models
        print("\nAvailable models:")
        for model in models:
            print(f"- {model.name}")
        
        # Try to use the first available Gemini model
        print("\n3. Looking for Gemini models...")
        model_name = None
        for model in models:
            if "gemini" in model.name.lower():
                model_name = model.name
                print(f"Found Gemini model: {model_name}")
                break
                
        if not model_name:
            logger.error("No Gemini models found")
            print("❌ API Test Failed: No Gemini models found")
            return False
            
        # Test the connection
        print(f"\n4. Testing connection with model: {model_name}")
        model = genai.GenerativeModel(model_name)
        print("Sending test request...")
        response = model.generate_content("Hello, this is a test.")
        
        if response:
            logger.info("Successfully received response from API")
            print("\nAPI Test Results:")
            print(f"✅ API configuration successful (using model: {model_name})")
            print("✅ Model connection successful")
            print("✅ Response received successfully")
            print(f"\nTest response: {response.text}")
            return True
        else:
            logger.error("No response received from API")
            print("❌ API Test Failed: No response received")
            return False
            
    except Exception as e:
        logger.error(f"API test failed: {str(e)}")
        print("\n❌ API Test Failed:")
        print(f"Error: {str(e)}")
        print("\nDetailed error information:")
        print(traceback.format_exc())
        print("\nTroubleshooting steps:")
        print("1. Verify your API key is correct")
        print("2. Check if you have enabled the Gemini API in Google Cloud Console")
        print("3. Ensure you have sufficient quota available")
        print("4. Verify your API key has the necessary permissions")
        print("5. Check your internet connection")
        print("6. Try using a different network if possible")
        return False

if __name__ == "__main__":
    test_api_connection() 