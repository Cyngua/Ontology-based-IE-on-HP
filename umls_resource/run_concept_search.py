'''
to run get-concepts-for-a-list-of-codes.py 
'''

import os
from dotenv import load_dotenv
import subprocess

# Load environment variables from .env file
load_dotenv()
apikey = os.getenv('UMLS_API_KEY')
folder = 'ulms_resource/search_result'
os.makedirs(folder, exist_ok=True)

if not apikey:
    raise ValueError("API key not found. Please set UMLS_API_KEY in your .env file.")

inputfile = 'input_1.txt'
outputfile = 'output_1.txt'
command = [
    'python', 'ulms_resource/get-concepts-for-a-list-of-codes.py',
    '-k', apikey,
    '-o', os.path.join(folder, outputfile),
    '-s', "SNOMEDCT_US",
    '-i', os.path.join(folder, inputfile)
]

# Execute the command
result = subprocess.run(command, capture_output=True, text=True)

# Output the result
if result.returncode == 0:
    print(f"Script executed successfully. Output saved to {outputfile}")
else:
    print(f"Script execution failed with return code {result.returncode}")
    print("Error message:", result.stderr)