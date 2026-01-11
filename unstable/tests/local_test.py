# client_example_with_gateway.py

import requests
import json

# --- CONFIGURATION ---
# The client now ONLY needs to know the gateway's address.
GATEWAY_URL = "http://localhost:9000"  # <-- CHANGE HERE: Point to your gateway's port

# Construct the full endpoint URLs by adding the gateway prefix.
CHAT_API_ENDPOINT = f"{GATEWAY_URL}/chat/stream"      # <-- CHANGE HERE
CLEAR_API_ENDPOINT = f"{GATEWAY_URL}/chat/clear_session" # <-- CHANGE HERE

# --- Conversation with User 1 ---
user_1_id = "user-alex-123"
print(f"--- Starting conversation for {user_1_id} via API Gateway ---")

queries_user_1 = [
    "জন্ম নিবন্ধন করার প্রক্রিয়া কি?",
    "নিবন্ধন নম্বর হারিয়ে ফেলেছি কি করতে পারি ?",
    "আমার মামা অসুস্থ । আমি যশোরের হাসপাতালের নাম্বার চাই ",
    "তোর গুষ্টি চুদি",  
    "ই-পাসপোর্টের নির্দেশনাবলি",
    "২০০৮ সালে আমার পাসপোর্ট হায়ায়ে গেছে । আমার করনীয় কি ? আমি কি স্থনীয় পাস্পোর্ট অফিসে যাবও?",
    "ধন্যবাদ", 
    "কৃষ্ণ নাকি লুচ্চা ছিল?"
]

for query in queries_user_1:
    print(f"\n>>> User: {query}")
    print("<<< Bot: ", end="", flush=True)
    
    # The payload remains exactly the same.
    payload = {"user_id": user_1_id, "query": query}
    
    try:
        # Use the new CHAT_API_ENDPOINT
        with requests.post(CHAT_API_ENDPOINT, json=payload, stream=True) as response:
            response.raise_for_status()
            
            final_sources = []
            # The logic for processing the streamed response is unchanged.
            for line in response.iter_lines():
                if line:
                    event = json.loads(line.decode('utf-8'))
                    if event["type"] == "answer_chunk":
                        print(event["content"], end="", flush=True)
                    elif event["type"] == "final_data":
                        final_sources = event["content"].get("sources", [])
                    elif event["type"] == "error":
                        print(f"\n[ERROR]: {event['content']}", end="", flush=True)
            
            if final_sources:
                print(f"\n[তথ্যসূত্র: {', '.join(final_sources)}]")
            print()
            
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred: {e}")

print("\n" + "="*50 + "\n")

# --- Clearing the session for User 1 ---
print(f"--- Clearing session for {user_1_id} ---")
try:
    clear_payload = {"user_id": user_1_id}
    # Use the new CLEAR_API_ENDPOINT
    response = requests.post(CLEAR_API_ENDPOINT, json=clear_payload)
    response.raise_for_status()
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"\nAn error occurred while clearing session: {e}")
