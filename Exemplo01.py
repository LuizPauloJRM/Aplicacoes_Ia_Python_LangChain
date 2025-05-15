import os
from decouple import config
from langchain_openai import OpenAI


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')
model = OpenAI()

question = input('O que deseja saber? ')

response = model.invoke(
    input=question,
)

print(response)
