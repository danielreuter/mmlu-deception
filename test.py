# %%
from dotenv import load_dotenv
from datasets import load_dataset

load_dotenv()

dataset = load_dataset("cais/mmlu")
# %%
import requests
def send_whoami_request():
    url = "https://huggingface.co/api/whoami-v2"
    headers = {
        "user-agent": "unknown/None; hf_hub/0.20.3; python/3.9.18",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
        "Connection": "keep-alive",
        "authorization": "Bearer xxx",
    }

    response = requests.get(url, headers=headers)

    # Print the response status code and content for inspection
    print("Status Code:", response.status_code)
    print("Response:", response.json())


send_whoami_request()
# %%
