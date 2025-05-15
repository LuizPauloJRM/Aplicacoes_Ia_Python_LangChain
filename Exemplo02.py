import os
from decouple import config

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

model = ChatOpenAI(
    model='gpt-4o',
)

question = input('O que deseja saber? ')

messages = [
    SystemMessage(
        content='Você é um assistente que fornece informações sobre figuras históricas.'
    ),
    HumanMessage(content=question),
    AIMessage(
        content='Alan Turing foi um matemático, lógico, criptógrafo e cientista da computação britânico.'
    ),
    HumanMessage(content='Até quantos anos ele viveu?'),
]

response = model.invoke(messages)

print(response)
print(response.content)
