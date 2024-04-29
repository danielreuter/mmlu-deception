from dotenv import load_dotenv
import instructor
from openai import OpenAI

load_dotenv()

client = instructor.from_openai(OpenAI())

def generate(schema, messages, model):
    print(messages)
    output = client.chat.completions.create(
        model=model,
        response_model=schema,
        messages=messages,
        temperature=0
    )
    print(output)
    return output