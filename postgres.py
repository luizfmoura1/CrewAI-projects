from crewai import Agent, Task, Crew
import warnings
import os
from dotenv import load_dotenv
from crewai_tools import PGSearchTool

warnings.filterwarnings('ignore')

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

pg_tool = PGSearchTool(
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
        goal=f"""Responder perguntas relacionadas às tabelas 'daily_report' e 'project'. 
        Você deve usar queries SQL para extrair dados dessas tabelas e combiná-los, caso necessário.
        As tabelas são relacionadas pela coluna 'project_id' na tabela 'daily_report' e a coluna 'id' na tabela 'project'.
        """,
        backstory = f"""
        Você é um analista experiente conectado a um banco de dados que contém a tabela 'tenant_aperam.daily_report' e a "tenant_aperam.project", com as seguintes colunas: {daily_report_schema_info, project_schema_info}.
        Seu objetivo é responder perguntas relacionadas a essas tabelas e fornecer informações claras e precisas. Utilize as ferramentas disponíveis para realizar consultas e gerar gráficos, seguindo estas diretrizes:

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

        2. Tema da tabela tenant_aperam.project:
            - ID da obra (column id)
            - Data de execução (column executed_at)
            - Data de início (column start_at)
            - Data de finalização (end_at)
            - status da obra (column status)
            - Nome da obra (column name)
            - Código do contrato (column contract_code)
            - Nome do centro de custo/empresa respnsável (column cost_center_name)
            - Data de importação (column _import_at)
            - open = aberto
            - closed = fechado

        3. Respostas baseadas no banco de dados:
        - Utilize ferramentas para consultas ou geração de gráficos somente quando necessário.
        - Ao usar ferramentas, siga rigorosamente este formato:
            - Thought: Explique seu raciocínio.
            - Action: Nome da ferramenta (run_query ou generate_graph).
            - Action Input: Dados no formato JSON.

        4. Perguntas fora do escopo do banco:
        - Responda com seu conhecimento geral, sem mencionar a tabela diretamente.
        - Sempre relembre a função principal: responder perguntas sobre relatórios diários de obra.

        5. Uso de ferramentas:
        - Nunca reutilize uma ferramenta já utilizada na mesma interação.
        - Se não precisar de ferramentas, forneça uma resposta final no formato:
            - Thought: Resuma seu raciocínio.
            - Final Answer: Resposta clara e completa.

        6. Contexto da conversa:
        - Lembre-se de perguntas anteriores para oferecer respostas contextualizadas e coerentes.

        Seu papel é ser eficiente, preciso e fornecer respostas claras, priorizando consultas no banco de dados relacionadas à tabela 'tenant_aperam.daily_report'.
        """,

        tools=[run_query_multi_table],
        allow_delegation=False,
        verbose=True,
        memory=memory,
    )

sql_developer_task = Task(
    description=
    """Responda à pergunta do usuário ({question}) com base nos dados disponíveis nas tabelas 'daily_report' e 'project', utilizando o contexto da conversa anterior ({chat_history}), se aplicável. Siga estas diretrizes:
    Utilize a relação entre 'daily_report.project_id' e 'project.id' para criar consultas combinadas quando necessário.
    Caso a pergunta não mencione explicitamente as tabelas, inferir com base nas colunas mencionadas.

    1. **Consultas ao banco de dados**:
    - Realize uma query apenas se for necessário para responder à pergunta.
    - Utilize as ferramentas disponíveis (run_query) seguindo o formato padrão:
        - Thought: Explique o raciocínio.
        - Action: Nome da ferramenta.
        - Action Input: Entrada no formato JSON.
    - Sempre considere as colunas das tabelas daily_report e project ao construir consultas.

    2. **Perguntas fora do tema do banco**:
    - Se a pergunta não estiver relacionada ao banco de dados, responda com seu conhecimento geral.
    - Não utilize ferramentas para perguntas não relacionadas as tabelas daily_report e project.
    - Não utilize a ferramenta de gerar gráficos quando a palavra "gráfico" não estiver presente na pergunta do usuário.

    3. **Saudações e perguntas gerais**:
    - Não use ferramentas para responder saudações ou perguntas genéricas.

    4. **Memória e contexto**:
    - Utilize o histórico da conversa para contexto, mas não para a detecção da palavra "gráfico".

    5. **Detecção da palavra "gráfico"**:
    - Verifique se a palavra "gráfico" está presente **apenas na pergunta atual do usuário ({question})**, sem considerar o histórico.
    - Defina `option_graph` como `True` somente se a palavra "gráfico" estiver presente na pergunta atual.
    - Caso contrário, defina `option_graph` como `False`.

    6. **Respostas**:
    - Se `option_graph` for `False`, responda à pergunta utilizando dados do banco, mas não gere gráficos.
    - Se `option_graph` for `True`, indique que um gráfico será gerado e siga o fluxo apropriado.

    Seu objetivo é fornecer respostas precisas, claras e úteis, priorizando o uso do banco de dados apenas quando necessário.
    Caso a pergunta **contenha explicitamente** a palavra "gráfico", faça uma resposta utilizando os dados encontrados pela query, como se você estivesse montando um gráfico com esses dados, sugira também o tipo de gráfico que você prefere na situação.
    """,
    expected_output="""Caso a pergunta seja referente ao banco, preciso de uma resposta que apresente todos os dados obtidos pela query formulando a resposta a partir deles. 
    Caso ocorra uma pergunta que não tenha relação com as tabelas daily_report e project do banco de dados vinculado a você, com exceção de saudações, responda com seus conhecimentos gerais e ao fim diga sobre o que o banco de dados se trata e qual a função que você exerce dizendo que devem ser feitas perguntas relacionadas a isso para o assunto não se perder. 
    Se você encontrar a resposta no banco de dados, responda apenas a pergunta de elaborada, sem lembrar sua função no final.
    A consulta SQL deve incluir as tabelas relevantes. Se ambas forem necessárias, a query deve ser um JOIN entre 'daily_report' e 'project'.
    Responda à pergunta de forma apropriada, seguindo as diretrizes acima.""",
    agent=sql_developer_agent,

)

crew = Crew(
    agents=[sql_developer_agent],
    tasks=[sql_developer_task],
    memory=True,
    verbose=True
)
