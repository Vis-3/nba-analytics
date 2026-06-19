import requests
import os

host = 'https://dbc-303xxxxxx.cloud.databricks.com'
token = ''
local_path = r'C:\Users\sansk\Downloads\nba.sqlite\nba.sqlite'
volume_path = '/Volumes/workspace/bronze_nba/raw_files/nba.sqlite'

headers = {'Authorization': f'Bearer {token}'}

print(f'File size: {os.path.getsize(local_path) / 1e9:.2f} GB')
print('Starting upload...')

with open(local_path, 'rb') as f:
    response = requests.put(
        f'{host}/api/2.0/fs/files{volume_path}',
        headers=headers,
        data=f,
        timeout=600
    )

print(f'Status: {response.status_code}')
print(f'Response: {response.text[:200]}')
