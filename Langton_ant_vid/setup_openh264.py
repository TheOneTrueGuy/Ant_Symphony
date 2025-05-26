import urllib.request
import os
import sys

def download_openh264():
    url = "https://github.com/cisco/openh264/releases/download/v1.8.0/openh264-1.8.0-win64.dll"
    dll_name = "openh264-1.8.0-win64.dll"
    
    # Get the Python directory where the DLL should be placed
    python_dir = os.path.dirname(sys.executable)
    target_path = os.path.join(python_dir, dll_name)
    
    print(f"Downloading OpenH264 to: {target_path}")
    try:
        urllib.request.urlretrieve(url, target_path)
        print("Successfully downloaded OpenH264!")
    except Exception as e:
        print(f"Error downloading OpenH264: {e}")
        return False
    
    return True

if __name__ == "__main__":
    download_openh264()
