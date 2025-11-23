import os
import json
from glob import glob

PASTA = "../output_noticias"

def normalize_ticker(t):
    """Remove .SA e =F e deixa tickers consistentes"""
    return t.replace(".SA", "").replace("=F", "").strip().upper()

def corrigir_jsons_output(pasta=PASTA):
    arquivos = glob(os.path.join(pasta, "evento_*.json"))
    print(f"🔍 Encontrados {len(arquivos)} arquivos JSON")

    for arq in arquivos:
        try:
            with open(arq, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            print(f"⚠ Erro ao abrir {arq}")
            continue

        alterou = False

        # --------------------------
        # 1) Corrigir campo "ativo"
        # --------------------------
        ativo = data.get("ativo", "")

        if ativo != "AMBOS":
            novo = normalize_ticker(ativo)
            if novo != ativo:
                data["ativo"] = novo
                alterou = True

        # --------------------------
        # 2) Corrigir impacto_d0_...
        # --------------------------
        chaves_corrigir = [k for k in data.keys() if k.startswith("impacto_d0_")]

        for k in chaves_corrigir:
            nome_ticker = normalize_ticker(k.replace("impacto_d0_", ""))
            novo_k = f"impacto_d0_{nome_ticker}"

            if novo_k != k:
                data[novo_k] = data[k]
                del data[k]
                alterou = True

        # --------------------------
        # 3) Corrigir retorno_no_dia e fechamento
        # --------------------------
        if isinstance(data.get("retorno_no_dia"), dict):
            novos = {}
            for k, v in data["retorno_no_dia"].items():
                novos[normalize_ticker(k)] = v
            data["retorno_no_dia"] = novos
            alterou = True

        if isinstance(data.get("fechamento"), dict):
            novos = {}
            for k, v in data["fechamento"].items():
                novos[normalize_ticker(k)] = v
            data["fechamento"] = novos
            alterou = True

        # --------------------------
        # 4) Se ativo == AMBOS → manter AMBOS, mas normalizar internos
        # --------------------------
        if data.get("ativo") == "AMBOS":
            # já normalizamos subestruturas acima
            pass

        # --------------------------
        # Salvar de volta só se mudou
        # --------------------------
        if alterou:
            with open(arq, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            print(f"✔ Corrigido: {os.path.basename(arq)}")

    print("\n🎉 Correção finalizada!")


if __name__ == "__main__":
    corrigir_jsons_output()
