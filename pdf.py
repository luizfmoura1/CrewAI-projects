from crewai import Agent, Task, Crew
import warnings
import os
from dotenv import load_dotenv
from crewai_tools import PDFSearchTool

warnings.filterwarnings('ignore')

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

# Initialize the tool with a specific PDF path for exclusive search within that document
search_tool = PDFSearchTool(
    pdf=r"C:\Oppem\crewAI\CrewAI-projects\Processo_Montagem_Carro.pdf",
    config=dict(
        llm=dict(
            provider="openai",
            config=dict(
                model="gpt-4",  # Modelo válido
                temperature=0.5,
                top_p=1,
            ),
        ),
        embedder=dict(
            provider="openai",
            config=dict(
                model="text-embedding-ada-002",  # Modelo de embedding
            ),
        ),
    )
)

search_pdf_agent = Agent(
    role="PDF search specialist",
    goal = "Seja preciso de acordo com o pdf, para responder a {question} do usuário",
    backstory="""Você é um especialista em processos de montagem de carros e tem acesso
          direto a um arquivo PDF detalhado sobre o tema.
          Sua função é consultar esse documento para fornecer informações precisas
          para responder a {question} do usuário.
          Você não deve utilizar seus conhecimentos gerais, deve responder apenas se encontrar a resposta dentro do documento.
          Forneça uma resposta precisa e específica para a pergunta do usuário, não traga a tona outros temas e processos que não tenham relação com a {question}""",
    allow_delegation=False,
    verbose=True
)

search_pdf_task = Task(
    description="""
        Realize uma consulta no arquivo PDF vinculado a você, que contém informações 
        detalhadas sobre o processo de montagem de carros. 
        A tarefa é buscar e fornecer uma resposta clara com base no conteúdo 
        do documento para responder a pergunta do usuário {question}. Certifique-se de responder apenas o que você encontrar no documento.
    """,
    expected_output="""
        Uma resposta detalhada e informativa que aborda a questão apresentada, 
        extraindo as informações diretamente do PDF. 
        A resposta deve ser clara, técnica quando necessário, e fácil de entender, 
        com foco em explicar o processo de montagem de carros.
        A resposta Final deve ser em forma de parágrafo e não em tópicos.
    """,
    tools=[search_tool],  
    agent=search_pdf_agent  
)

crew = Crew(
    agents=[search_pdf_agent],
    tasks=[search_pdf_task],
    verbose=True,
    memory=True
)



print("Chat iniciado. Pergunte sobre o processo de montagem de carros. Digite 'quit' para sair.\n")

while True:
    question = input("User: ").strip() 
    if question.lower() == "quit":  #
        print("Encerrando o chatbot. Até logo!")
        break

    try:
        # Passar a pergunta para o Crew
        result = crew.kickoff(inputs={"query": question, "question": question})
        print(f"Bot: {result}\n")
    except Exception as e:
        print(f"Erro: {str(e)}. Tente novamente.\n")
