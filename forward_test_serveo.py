import subprocess
import sys
import time
import requests
import socket
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def verify_tcp_connection(host, port):
    try:
        sock = socket.create_connection((host, port), timeout=10)
        sock.close()
        return True
    except Exception as e:
        print(f"TCP connection failed: {str(e)}")
        return False

def check_local_server(port):
    try:
        # First try socket connection
        sock = socket.create_connection(("localhost", port), timeout=5)
        sock.close()
        
        # Then try HTTP request
        response = requests.get(f"http://localhost:{port}/available_models", timeout=5)
        return response.status_code == 200
    except:
        return False

def forward_to_serveo(local_port=5001, subdomain="scamba-test-server"):
    if not check_local_server(local_port):
        print(f"Error: Local server not running on port {local_port}")
        print("Please start the test server first using run_test_server.bat")
        sys.exit(1)

    command = f"ssh -o ServerAliveInterval=60 -R {subdomain}:80:127.0.0.1:{local_port} serveo.net"
    
    try:
        print(f"Starting port forwarding for 127.0.0.1:{local_port} to {subdomain}.serveo.net")
        process = subprocess.Popen(command, shell=True)
        
        print("Waiting for tunnel to establish...")
        time.sleep(15)  # Increased initial wait
        
        session = create_session()
        retries = 0
        max_retries = 5  # Increased max retries
        
        while True:
            try:
                # Verify local server first
                if not check_local_server(local_port):
                    print("Warning: Local server not responding...")
                    time.sleep(5)
                    continue

                # Verify TCP connection to serveo
                if not verify_tcp_connection(f"{subdomain}.serveo.net", 443):
                    print("Warning: Cannot establish TCP connection to Serveo...")
                    time.sleep(5)
                    continue
                
                # Check forwarded URL with increased timeout
                response = session.get(
                    f"https://{subdomain}.serveo.net/available_models",
                    timeout=20,
                    headers={'Connection': 'close'}  # Prevent keep-alive issues
                )
                
                if response.status_code == 200:
                    print(f"Forwarding working correctly: {time.strftime('%H:%M:%S')} - Status: {response.status_code}")
                    retries = 0
                else:
                    print(f"Warning: Status {response.status_code} - {response.text}")
                    retries += 1
                
                if retries >= max_retries:
                    print("Error: Too many failed attempts. Restarting tunnel...")
                    process.terminate()
                    time.sleep(10)
                    process = subprocess.Popen(command, shell=True)
                    retries = 0
                    time.sleep(15)
                
            except requests.exceptions.RequestException as e:
                print(f"Connection error: {str(e)}")
                retries += 1
                time.sleep(5)  # Wait before retry
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                retries += 1
                time.sleep(5)
            
            time.sleep(20)  # Increased check interval
            
    except KeyboardInterrupt:
        print("\nStopping port forwarding...")
        process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    forward_to_serveo()
