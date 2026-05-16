from dotenv import load_dotenv
from search import search_prompt

load_dotenv()

def main():
    while True:
        q = input("PERGUNTA: ").strip()
        if not q or q.lower() in {"sair", "exit", "quit"}:
            break
        resposta = search_prompt(q)
        print(f"RESPOSTA: {resposta}\n")

if __name__ == "__main__":
    main()