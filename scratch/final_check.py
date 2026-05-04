import requests
import json

def test_api():
    url = "http://localhost:8000/api/v1/chat"
    payload = {
        "session_id": "test_session",
        "message": "What is the punishment for murder under section 302 IPC?"
    }
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Raw Response: {response.text}")

if __name__ == "__main__":
    test_api()
