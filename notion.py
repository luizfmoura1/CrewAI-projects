from crewai import Agent, Task, Crew
import warnings
import os
from dotenv import load_dotenv
from notion_client import Client
from crewai_tools import BaseTool
from pydantic import PrivateAttr


warnings.filterwarnings('ignore')

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
NOTION_API_KEY = os.getenv('NOTION_API_KEY')
DATABASE_ID = os.getenv('DATABASE_ID')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

class NotionSearchTool(BaseTool):
    name: str = "Notion Search Tool"
    description: str = "Uma ferramenta para buscar informações específicas em um banco de dados do Notion."

    # Declara os atributos como privados
    _client: Client = PrivateAttr()
    _database_id: str = PrivateAttr()

    def __init__(self, notion_api_key: str, database_id: str):
        super().__init__()
        self._client = Client(auth=notion_api_key)
        self._database_id = database_id

    def _run(self, query: str) -> str:
        """Realiza uma busca no Notion com base no conteúdo do banco de dados."""
        try:
            results = self._client.databases.query(
                database_id=self._database_id,
                filter={
                    "property": "Name",  # Substitua por uma propriedade válida no Notion
                    "rich_text": {"contains": query}
                }
            )
            pages = [
                page["properties"]["Name"]["title"][0]["plain_text"]
                for page in results["results"]
            ]
            return "\n".join(pages) if pages else "Nenhuma informação encontrada no Notion."
        except Exception as e:
            return f"Erro ao buscar no Notion: {e}"



notion_search_tool = NotionSearchTool(notion_api_key=NOTION_API_KEY, database_id=DATABASE_ID)


search_notion_agent = Agent(
    role="Especialista em Busca no Notion",
    goal="Seja preciso de acordo com as informações do Notion, para responder a {question} do usuário",
    backstory="""Você é um especialista no processo de criação de RDOs na plataforma Opus e tem acesso
          direto a um banco de dados no Notion que contém informações detalhadas sobre o tema.
          Sua função é consultar esse banco de dados para fornecer informações precisas
          para responder a {question} do usuário.
          Você não deve utilizar seus conhecimentos gerais, deve responder apenas se encontrar a resposta dentro do conteúdo do Notion.
          Forneça uma resposta precisa e específica para a pergunta do usuário, não traga à tona outros temas e processos que não tenham relação com a {question}.""",
    allow_delegation=False,
    verbose=True
)

# Tarefa adaptada para busca no Notion
search_notion_task = Task(
    description="""
        Realize uma busca no banco de dados do Notion vinculado a você, que contém informações 
        detalhadas sobre o processo de criação de RDOs na plataforma Opus. 
        A tarefa é buscar e fornecer uma resposta clara com base no conteúdo 
        encontrado no Notion para responder à pergunta do usuário {question}.
        Certifique-se de responder apenas o que você encontrar no banco de dados.
    """,
    expected_output="""
        Uma resposta detalhada e informativa que aborda a questão apresentada, 
        extraindo as informações diretamente do banco de dados do Notion. 
        A resposta deve ser clara, técnica quando necessário, e fácil de entender, 
        com foco em explicar o processo de montagem de carros.
        A resposta final deve ser em forma de parágrafo e não em tópicos.
    """,
    tools=[notion_search_tool],
    agent=search_notion_agent
)

# Crew configurado para busca no Notion
crew = Crew(
    agents=[search_notion_agent],
    tasks=[search_notion_task],
    verbose=True,
    memory=True
)


# Loop do chatbot
print("Chatbot Notion iniciado. Digite 'quit' para sair.\n")
while True:
    question = input("User: ").strip()
    if question.lower() == "quit":
        print("Encerrando o chatbot. Até logo!")
        break

    result = crew.kickoff(inputs={"query": question, "question": question})
    print(f"Bot: {result}\n")
