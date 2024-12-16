from crewai import Agent, Task, Crew
import warnings
import os
from dotenv import load_dotenv
from crewai_tools import PDFSearchTool

# Ignorar warnings
warnings.filterwarnings('ignore')

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

tool = PDFSearchTool(
    pdf="C:/Oppem/crewAI/CrewAI-projects/Process_Montagem_Carro.pdf",
    config=dict(
        llm=dict(
            provider="openai",
            config=dict(
                model="gpt-4o-mini",
                temperature=0.5,
                top_p=1,
            ),
        ),
        embedder=dict(
            provider="openai",
            config=dict(
                model="gpt-4o-mini",
            ),
        ),
    )
)

# Configuração do agente
search_pdf_agent = Agent(
    role="PDF search",
    goal="Encontrar a resposta coerente à pergunta do usuário {question} com base no conteúdo do PDF",
    backstory="""Você está conectado a um arquivo PDF. Seu trabalho é buscar informações diretamente
    no texto do PDF para responder à pergunta do usuário. Caso não encontre uma resposta clara, você
    deve explicar o tema principal do PDF com base no texto extraído.""",
    tools=[tool],
    allow_delegation=False,
    verbose=True
)


# Configuração da tarefa
search_task = Task(
    description=(
        """Receba a pergunta do usuário: {question}
        Busque informações no texto do PDF que respondam diretamente ou algo relacionado à pergunta.
        Se encontrar, forneça a resposta exata. Caso contrário, descreva o tema principal do PDF."""
),

    expected_output=(
        "Uma resposta clara e coerente para a pergunta do usuário: {question}"
        "Se a resposta não for encontrada, responda com algo que faça sentido relacionar com a pergunta do usuário"
        "Se a pergunta for totalmente disconexa do tema do arquivo, você deve responde que não consegue responder perguntas que não sejam do tema do banco "
    ),
    agent=search_pdf_agent
)

# Configuração da equipe
crew = Crew(
    agents=[search_pdf_agent],
    tasks=[search_task],
    verbose=True
)

# Loop para conversação
while True:
    # Entrada do usuário
    question = input("Qual a sua pergunta (digite 'quit' para sair): ").strip()


    # Verificar se o usuário deseja sair
    if question.lower() == "quit":
        print("Encerrando o sistema. Até mais!")
        break

    try:
        result = crew.kickoff(inputs={"question": question})
        print("\nResposta do agente:")
        print(result.raw)
    except Exception as e:
        print(f"Ocorreu um erro durante a execução: {e}")
        print("Certifique-se de que o PDF está acessível e o formato da entrada está correto.")
