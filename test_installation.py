import sys
import importlib

def test_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} is installed and working")
        return True
    except ImportError as e:
        print(f"❌ {module_name} is NOT installed or has an error: {str(e)}")
        return False

def main():
    print("Python version:", sys.version)
    print("\nTesting required packages:")
    
    # List of required packages
    packages = [
        'flask',
        'werkzeug',
        'whisper',
        'google.generativeai',
        'moviepy',
        'gtts',
        'numpy',
        'ratelimit'
    ]
    
    all_installed = True
    for package in packages:
        if not test_import(package):
            all_installed = False
    
    if all_installed:
        print("\n✅ All packages are installed successfully!")
    else:
        print("\n❌ Some packages are missing or have errors. Please check the errors above.")

if __name__ == "__main__":
    main() 