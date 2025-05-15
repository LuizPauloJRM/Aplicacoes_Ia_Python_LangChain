# Aplicacoes_Ia_Python_LangChain
LangChain é framework/ecossistema de código aberto para o desenvolvimento de aplicativos com grandes modelos de linguagem (LLMs). Disponível em bibliotecas Python e Java, as ferramentas e APIs do LangChain simplificam o processo de criação de aplicativos baseados em LLM, como chatbots e agentes virtuais.
## O que é LangChain?

https://python.langchain.com/docs/introduction/

LangChain é framework/ecossistema de código aberto para o desenvolvimento de aplicativos com grandes modelos de linguagem (LLMs). Disponível em bibliotecas Python e Java, as ferramentas e APIs do LangChain simplificam o processo de criação de aplicativos baseados em LLM, como chatbots e agentes virtuais.

**O Django da Inteligência Artificial!**

O ecossistema do LangChain está em acelerado crescimento e já conta com diversos pacotes: https://python.langchain.com/docs/integrations/platforms/

- langchain e langchain_core
- langchain_community
- langchain_openai
- langchain_aws
- langchain_chroma

## Vamos para a prática…

*Instalação dos pacotes necessários*

```bash
pip install langchain
pip install langchain_openai
pip install python-decouple
```

*Variável de ambiente da OpenAI*

*.env*

```bash
OPENAI_API_KEY='SUA CHAVE DE API'
```

*Exemplo 01: invoke simples de LLM*

```python
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

```

*Exemplo 02: usando chat messages*

```python
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

```

*Exemplo 03: usando prompt templates*

```python
import os
from decouple import config

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

model = ChatOpenAI(
    model='gpt-4o',
)

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content='Você deve responder baseado em dados geográficos de regiões do Brasil.'
        ),
        HumanMessagePromptTemplate.from_template(
            template='Por favor, me fale sobre a região {regiao}.'
        ),
        AIMessage(
            content='Claro, vou começar coletando informações sobre a região e analisando os dados disponíveis.'
        ),
        HumanMessage(
            content='Certifique-se de incluir dados demográficos.'
        ),
        AIMessage(
            content='Entendido. Aqui estão os dados:'
        ),
    ]
)

regiao = input('Sobre qual região deseja saber? ')

prompt = chat_template.format_messages(regiao=regiao)
print(prompt)

response = model.invoke(prompt)
print(response)
print(response.content)

```

*Exemplo 04: tools nativas*

*Instalação de libs:*

```python
pip install langchain_community
pip install langchain_experimental
pip install duckduckgo-search
pip install wikipedia
```

*04.py*

```python
# DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchRun

ddg_search = DuckDuckGoSearchRun()

search_result = ddg_search.run('Quem foi Alan Turing?')
print(search_result)

# PythonREPL
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()
result = python_repl.run('print(5 + 5)')
print(result)

# WikipediaQueryRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        lang='pt'
    )
)

wikipedia_results = wikipedia.run('Quem foi Alan Turing?')
print(wikipedia_results)

```

*Exemplo 05: agent IPCA com banco de dados SQL*

*instalação de libs:*

```python
pip install beautifulsoup4
```

*ipca_scraper.py*

```python
import requests
import sqlite3
from bs4 import BeautifulSoup

url = 'https://www.idealsoftwares.com.br/indices/ipca_ibge.html'

response = requests.get(url)
html_content = response.content

soup = BeautifulSoup(html_content, 'html.parser')

table = soup.find_all(
    name='table',
    attrs={'class': 'table table-bordered table-striped text-center'},
)[1]

ipca_data = []
for row in table.find_all('tr')[1:]:
    cols = row.find_all('td')
    if cols:
        month_year = cols[0].text.strip()
        value = cols[1].text.strip()\
                            .replace(',', '.')\
                            .replace(' ', '').replace('\n', '')
        if value:
            month, year = month_year.split('/')
            ipca_data.append((float(value), month, int(year)))

conn = sqlite3.connect('ipca.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS IPCA (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    value REAL,
    month TEXT,
    year INTEGER,
    UNIQUE(month, year)
)
''')

for data in ipca_data:
    value, month, year = data
    cursor.execute('''
    INSERT OR IGNORE INTO IPCA (value, month, year)
    VALUES (?, ?, ?)
    ''', (value, month, year))

conn.commit()
conn.close()

print('Dados históricos do IPCA salvos com sucesso!')

```

*05.py*

```python
import os
from decouple import config

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

model = ChatOpenAI(
    model='gpt-4o',
)

db = SQLDatabase.from_uri('sqlite:///ipca.db')

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
)
system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

prompt = '''
Use as ferrmentas necessárias para responder perguntas relacionadas ao histórico de IPCA ao longo dos anos.
Responda tudo em português brasileiro.
Perguntas: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)

question = input('O que deseja saber sobre IPCA? ')

output = agent_executor.invoke({
    'input': prompt_template.format(q=question),
})

print(output.get('output'))

```

*Exemplo 06: agent de estoque com banco de dados SQL*

```python
import os
from decouple import config

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

model = ChatOpenAI(
    model='gpt-4o',
)

db = SQLDatabase.from_uri('sqlite:///db.sqlite3')

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
)
system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

prompt = '''
Use as ferrmentas necessárias para responder perguntas relacionadas ao
estoque de produtos. Você fornecerá insights sobre produtos, preços, 
reposição de estoque e relatórios conforme solitiado pelo usuário.
A resposta final deve ter uma formatação amigável
de visualização para o usuário.
Pergunta: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)

question = input('O que deseja saber sobre o estoque? ')

output = agent_executor.invoke({
    'input': prompt_template.format(q=question),
})

print(output.get('output'))

```