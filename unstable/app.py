# app.py

import gradio as gr
import requests # To make HTTP requests to your API
import json     # To parse the streaming response from your API
import uuid     # To create a unique session ID for the conversation

# --- 1. API and Session Configuration ---
# The Gradio client only needs to know the address of the API Gateway.
# All backend logic is now handled by the separate govtchat service.

# IMPORTANT: Make sure this URL points to your running API Gateway.
GATEWAY_URL = "http://114.130.116.74" 
CHAT_API_ENDPOINT = f"{GATEWAY_URL}/govtchat/chat/stream"

# Create a single, unique session ID for the entire duration of this Gradio app instance.
# The backend service will use this to keep track of the conversation history.
SESSION_ID = f"gradio-session-{uuid.uuid4()}"
print(f"Gradio App is running. All conversations will use Session ID: {SESSION_ID}")


# --- 2. Define the Core Chatbot Function (API Client) ---
# This function is the bridge between Gradio's UI and our backend API.
# It remains unchanged as its logic is independent of the display language.
def predict(message: str, history: list):
    """
    The main prediction function that calls the backend chat API.

    Args:
        message (str): The user's input message from the Gradio UI.
        history (list): The chat history managed by Gradio (we ignore this and rely on the backend's session).

    Yields:
        str: A stream of strings that builds the chatbot's response in the UI.
    """
    
    # Immediately yield a "Thinking..." placeholder for better user experience.
    yield "⌛ প্রক্রিয়াকরণ চলছে..." # Changed to Bangla

    # This list will accumulate the full response as it's streamed from the API.
    full_answer_list = []
    
    # This is the data we will send in our POST request to the backend.
    payload = {
        "user_id": SESSION_ID,
        "query": message
    }

    try:
        # Make a POST request to the streaming API endpoint.
        with requests.post(CHAT_API_ENDPOINT, json=payload, stream=True, timeout=300) as response:
            # Raise an exception if the API returns an error status code (e.g., 404, 500).
            response.raise_for_status()
            
            # Iterate over the streaming response line by line.
            for line in response.iter_lines():
                if line:
                    # Each line from the service is a JSON object representing an event.
                    event = json.loads(line.decode('utf-8'))
                    
                    if event["type"] == "answer_chunk":
                        # Append the new piece of the answer to our list.
                        full_answer_list.append(event["content"])
                        # Yield the joined list to update the Gradio UI in real-time.
                        yield "".join(full_answer_list)
                    
                    elif event["type"] == "final_data":
                        # The API signals that the main answer is complete and sends sources.
                        sources = event["content"].get("sources", [])
                        if sources:
                            # Format the sources and append them to the final answer.
                            source_str = "\n\n---\n*তথ্যসূত্র:* " + ", ".join(sources) # Changed to Bangla
                            full_answer_list.append(source_str)
                            # Yield the final, complete message with sources.
                            yield "".join(full_answer_list)

                    elif event["type"] == "error":
                        # If the backend sends a specific error event, display it.
                        yield f"ত্রুটি দেখা দিয়েছে: {event['content']}" # Changed to Bangla
                        return # Stop processing on error

    except requests.exceptions.RequestException as e:
        # Catch network-related errors (e.g., cannot connect to the server).
        print(f"An API connection error occurred: {e}")
        yield f"দুঃখিত, আমি চ্যাট সার্ভিসের সাথে সংযোগ স্থাপন করতে পারিনি। অনুগ্রহ করে নিশ্চিত করুন যে ব্যাকএন্ড চলছে। ত্রুটি: {e}"
    except Exception as e:
        # Catch any other unexpected errors.
        print(f"An unexpected error occurred in the predict function: {e}")
        yield "একটি অপ্রত্যাশিত ত্রুটি ঘটেছে। আবার চেষ্টা করুন."


# --- 3. Configure and Launch the Gradio UI ---
# This section has been updated with the new Bangla titles and examples.
demo = gr.ChatInterface(
    fn=predict,
    title="বাংলা VPA - আপনার ভার্চুয়াল সহায়ক",
    description="সরকারি সেবা সম্পর্কে জিজ্ঞাসা করুন, এবং আমি আপনাকে সঠিক তথ্য দিয়ে সহায়তা করব।",
    examples=[
        ["জন্ম নিবন্ধন করার প্রক্রিয়া কি?"],
        ["ট্রেড লাইসেন্স কিভাবে পাবো?"],
        ["পাসপোর্ট করার জন্য কি কি কাগজপত্র প্রয়োজন?"]
    ],
    cache_examples=False,
)

if __name__ == "__main__":
    # Launch the Gradio app. It will be accessible on your local network.
    demo.launch(server_name="0.0.0.0")