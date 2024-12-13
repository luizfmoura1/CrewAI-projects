from crewai import Agent, Task, Crew
import warnings
import os
from dotenv import load_dotenv
import PyPDF2  # Biblioteca para manipular PDFs

# Ignorar warnings
warnings.filterwarnings('ignore')

# Carregar variáveis de ambiente
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

# Ler o PDF
pdf_path = "Processo_Montagem_Carro.pdf"
with open(pdf_path, "rb") as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

# Configuração do agente
search_pdf_agent = Agent(
    role="PDF search",
    goal="Encontrar a resposta coerente à pergunta do usuário com base no conteúdo do PDF",
    backstory="""Você está conectado a um arquivo PDF. Seu trabalho é buscar informações diretamente
    no texto do PDF para responder à pergunta do usuário. Caso não encontre uma resposta clara, você
    deve explicar o tema principal do PDF com base no texto extraído.""",
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
    print("Texto extraído do PDF (primeiros 500 caracteres):")
    print(pdf_text[:500])  # Mostra os primeiros 500 caracteres do texto


    # Verificar se o usuário deseja sair
    if question.lower() == "quit":
        print("Encerrando o sistema. Até mais!")
        break

    # Executar o fluxo
    result = crew.kickoff(inputs={"question": question, "pdf_content": pdf_text})

    # Exibir o resultado no terminal
    print("\nResposta do agente:")
    print(result.raw)
