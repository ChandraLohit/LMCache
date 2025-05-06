#!/usr/bin/env python3
import requests
import json
import time

# OpenAI API compatible endpoint
url = "http://localhost:8000/v1/chat/completions"

# Define headers with dummy API key
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy-key"
}

# First request - this will be stored in Membrain
first_payload = {
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
}

# Send first request
print("Sending first request...")
start_time = time.time()
first_response = requests.post(url, headers=headers, data=json.dumps(first_payload))
first_duration = time.time() - start_time

print(f"First request took {first_duration:.2f} seconds")
print("Response:", first_response.json())

# Wait a moment
print("\nWaiting 5 seconds before sending the second request...")
time.sleep(5)

# Second request with the same prompt - this should be faster due to KV cache retrieval
print("Sending second request with same prompt...")
start_time = time.time()
second_response = requests.post(url, headers=headers, data=json.dumps(first_payload))
second_duration = time.time() - start_time

print(f"Second request took {second_duration:.2f} seconds")
print("Response:", second_response.json())

# Calculate and display speedup
if first_duration > 0 and second_duration > 0:
    speedup = first_duration / second_duration
    print(f"\nSpeedup from using Membrain KV cache: {speedup:.2f}x")