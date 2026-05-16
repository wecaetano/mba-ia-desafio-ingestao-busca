# Desafio MBA Engenharia de Software com IA - Full Cycle

RAG (Retrieval Augmented Generation) sobre PDF usando LangChain, PostgreSQL + pgVector e OpenAI.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- Chave de API da OpenAI ([platform.openai.com](https://platform.openai.com))

## Configuração

1. Clone o repositório e entre na pasta do projeto.

2. Crie o arquivo `.env` a partir do exemplo e preencha sua chave:
   ```bash
   cp .env.example .env
   ```
   Edite `.env` e defina `OPENAI_API_KEY=sk-...`.

3. Crie e ative o ambiente virtual:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

### 2. Ingerir o PDF

Certifique-se de que o arquivo `document.pdf` está na raiz do projeto e que `PDF_PATH=document.pdf` está no `.env`.

```bash
python src/ingest.py
```

### 3. Rodar o chat

```bash
python src/chat.py
```

Exemplo de uso:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Qual a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: sair
```

## Variáveis de ambiente

| Variável | Descrição |
|---|---|
| `OPENAI_API_KEY` | Chave da API OpenAI (obrigatório) |
| `OPENAI_EMBEDDING_MODEL` | Modelo de embeddings (padrão: `text-embedding-3-small`) |
| `OPENAI_CHAT_MODEL` | Modelo de chat (padrão: `gpt-4o-mini`) |
| `DATABASE_URL` | URL de conexão PostgreSQL |
| `PG_VECTOR_COLLECTION_NAME` | Nome da collection no pgVector |
| `PDF_PATH` | Caminho para o arquivo PDF |

## Stack

- **LangChain** — framework RAG
- **OpenAI** — embeddings (`text-embedding-3-small`) e LLM (`gpt-4o-mini`)
- **PostgreSQL + pgVector** — armazenamento vetorial
- **Docker Compose** — infraestrutura local
