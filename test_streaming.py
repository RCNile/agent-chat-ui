#!/usr/bin/env python3
"""
Simple test script to check the streaming response format
"""

import requests
import json

def test_streaming_endpoint():
    """Test the streaming endpoint to see what format is being returned"""
    url = 'http://localhost:2024/threads/test-thread/runs/stream'
    data = {
        'input': {
            'messages': [
                {
                    'id': 'test-1',
                    'type': 'human',
                    'content': [{'type': 'text', 'text': 'hello'}]
                }
            ]
        }
    }

    try:
        print("Testing streaming endpoint...")
        response = requests.post(url, json=data, stream=True)
        print(f'Status: {response.status_code}')
        print(f'Headers: {dict(response.headers)}')
        print('\nStreaming response:')
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode()
                print(f'Raw line: {decoded_line}')
                
                # Try to parse the JSON data
                if decoded_line.startswith('data: '):
                    json_str = decoded_line[6:]  # Remove 'data: ' prefix
                    try:
                        parsed_data = json.loads(json_str)
                        print(f'Parsed JSON: {json.dumps(parsed_data, indent=2)}')
                    except json.JSONDecodeError as e:
                        print(f'Failed to parse JSON: {e}')
                
                print('---')
                
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_streaming_endpoint()
