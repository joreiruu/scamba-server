import subprocess
import sys
import time

def forward_to_serveo(local_port=5001, subdomain="scambaserver"):
    command = f"ssh -R {subdomain}:80:localhost:{local_port} serveo.net"
    
    try:
        print(f"Starting port forwarding for localhost:{local_port} to {subdomain}.serveo.net")
        process = subprocess.Popen(command, shell=True)
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping port forwarding...")
        process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    forward_to_serveo()
