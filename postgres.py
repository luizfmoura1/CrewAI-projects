from crewai import Agent, Task, Crew
import warnings
import os
import psycopg2
from dotenv import load_dotenv
from crewai_tools import PGSearchTool

warnings.filterwarnings('ignore')

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'


try:
    connection = psycopg2.connect(
        dbname="gerdau",
        user='luiz',
        password='CgvQTiyXXEN7xSnsMHBkT5NW2MaxtC',
        host='localhost',
        port= 5432
    )
    print("Conexão bem-sucedida!")
    connection.close()
except Exception as e:
    print(f"Erro ao conectar ao banco de dados: {e}")


connection_pg_tool = PGSearchTool(
    db_uri='postgresql://gerdau:gerdau@localhost:6432/gerdau', 
    table_name='daily_report',
    config=dict(
        llm=dict(
            provider="openai", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="gpt-4o-mini",
                temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="openai", # or openai, ollama, ...
            config=dict(
                model="text-embedding-ada-002",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
sql_developer_agent = Agent(
    role='Postgres analyst senior',
    goal=f"""Responder perguntas relacionadas exclusivamente à tabela 'daily_report'. 
    Você deve usar queries SQL para extrair dados dessa tabela e fornecer respostas baseadas nas informações disponíveis.
    """,
    backstory=f"""
    Você é um analista experiente conectado a um banco de dados que contém a tabela 'tenant_aperam.daily_report', com as seguintes colunas: 
    'daily_report_schema_info'.
    Seu objetivo é responder perguntas relacionadas a essa tabela e fornecer informações claras e precisas. Utilize as ferramentas disponíveis para realizar consultas, seguindo estas diretrizes:

    1. Tema principal da tabela tenant_aperam.daily_report:
    - Relatórios diários de obra, com as seguintes colunas:
        - ID do relatório (column id)
        - Data de execução (column executed_at)
        - Data de criação (column created_at)
        - ID da obra (column project_id)
        - Data de aprovação (column approved_at)
        - Número sequencial (column sequence)
        - Usuário criador (column user_username)
        - Início e término do almoço (columns lunch_start_time, lunch_end_time)
        - Início e término do expediente (columns work_start_time, work_end_time)
        - Comentários (column comment)
        - Status do relatório (column status)
        - Nome do empreiteiro (column builder_name)
        - Data de assinatura do empreiteiro (column builder_signed_at)
        - Quantidade de revisões (column revision_number)
        - Data de importação (column _import_at)
        - 'approved' = aprovado
        - 'in_review' = em análise
        - 'in_approver' = em aberto

    2. Respostas baseadas no banco de dados:
    - Utilize ferramentas para consultas SQL somente quando necessário.
    - Ao usar ferramentas, siga rigorosamente este formato:
        - Thought: Explique seu raciocínio.
        - Action Input: Dados no formato JSON.

    3. Perguntas fora do escopo do banco:
    - Responda com seu conhecimento geral, sem mencionar a tabela diretamente.
    - Sempre relembre a função principal: responder perguntas sobre relatórios diários de obra.

    4. Uso de ferramentas:
    - Nunca reutilize uma ferramenta já utilizada na mesma interação.
    - Se não precisar de ferramentas, forneça uma resposta final no formato:
        - Thought: Resuma seu raciocínio.
        - Final Answer: Resposta clara e completa.

    5. Contexto da conversa:
    - Lembre-se de perguntas anteriores para oferecer respostas contextualizadas e coerentes.

    Seu papel é ser eficiente, preciso e fornecer respostas claras, priorizando consultas no banco de dados relacionadas à tabela 'tenant_aperam.daily_report'.
    """,
    allow_delegation=False,
    verbose=True,
    max_iter=2
)


sql_developer_task = Task(
    description=
    """Responda à pergunta do usuário ({question}) com base nos dados disponíveis na tabela 'daily_report', Siga estas diretrizes:
    Caso a pergunta não mencione explicitamente a tabela, infira com base nas colunas mencionadas.

    1. **Consultas ao banco de dados**:
    - Realize uma query apenas se for necessário para responder à pergunta.
    - Utilize as ferramentas disponíveis (run_query) seguindo o formato padrão:
        - Thought: Explique o raciocínio.
        - Action: Nome da ferramenta.
        - Action Input: Entrada no formato JSON.
    - Sempre considere as colunas da tabela daily_report ao construir consultas.

    2. **Perguntas fora do tema do banco**:
    - Se a pergunta não estiver relacionada ao banco de dados, responda com seu conhecimento geral.
    - Não utilize ferramentas para perguntas não relacionadas à tabela daily_report.

    3. **Saudações e perguntas gerais**:
    - Não use ferramentas para responder saudações ou perguntas genéricas.

    4. **Memória e contexto**:
    - Utilize o histórico da conversa para contexto, mas não para inferir informações ausentes na pergunta atual.

    5. **Respostas**:
    - Responda à pergunta utilizando dados do banco, mas não gere gráficos.
    - Caso os dados necessários estejam indisponíveis, explique isso claramente ao usuário.

    Seu objetivo é fornecer respostas precisas, claras e úteis, priorizando o uso do banco de dados apenas quando necessário.
    Caso a pergunta esteja relacionada ao banco de dados, forneça uma resposta detalhada baseada nos dados encontrados pela query.
    """,
    expected_output="""Caso a pergunta seja referente ao banco, forneça uma resposta que apresente todos os dados obtidos pela query formulada.
    Caso ocorra uma pergunta que não tenha relação com a tabela daily_report do banco de dados vinculado a você, com exceção de saudações, responda com seus conhecimentos gerais e ao fim explique que o banco de dados se refere a relatórios diários e que você está disponível para responder perguntas relacionadas a ele.
    Se você encontrar a resposta no banco de dados, responda apenas à pergunta de forma direta e clara, sem lembrar sua função no final.
    A consulta SQL deve incluir apenas a tabela daily_report, sem necessidade de JOIN com outras tabelas.
    Responda à pergunta de forma apropriada, seguindo as diretrizes acima.""",
    agent=sql_developer_agent,
    tools=[connection_pg_tool],
)


crew = Crew(
    agents=[sql_developer_agent],
    tasks=[sql_developer_task],
    memory=True,
    verbose=True
)

print("Chat iniciado. Pergunte sobre o processo de criação de RDOs. Digite 'quit' para sair.\n")

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