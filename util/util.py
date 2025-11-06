import re

# remover caracteres especiais de uma string e numeros

def limpar_texto(texto: str) -> str:
    texto_limpo = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()
    return texto_limpo

if __name__ == "__main__":
    with open("../input.txt", "r", encoding="utf-8") as file:
        conteudo = file.read()


    print(limpar_texto(conteudo))

