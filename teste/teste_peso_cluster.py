from src.clusters_utils import obter_impacto_real

# Função simples para testar impacto REAL em previsões float
def aplicar_correcao_simples(prev, impacto_real, k, meia_vida):
    alpha_k = meia_vida ** k
    impacto = impacto_real if k == 0 else impacto_real * alpha_k
    final = prev * (1 + impacto)
    return final, impacto


def run_test_frase(frase: str):

    print("\n===============================================")
    print(f"🔍 Testando frase: {frase}")
    print("===============================================\n")

    info = obter_impacto_real([frase])

    cluster = info["cluster"]
    impacto_real = info["impacto_real"]
    motivo_base = info["motivo_referência"]
    similaridade = info["similaridade"]

    print("📌 Resultado da classificação:")
    print(f" - Cluster detectado : {cluster}")
    print(f" - Impacto real médio: {impacto_real*100:.2f}%")
    print(f" - Motivo referência : {motivo_base}")
    print(f" - Similaridade      : {similaridade:.3f}\n")

    previsoes = [0.01, 0.012, 0.009, 0.008]
    meia_vida = 0.6

    print("📈 Aplicando impacto real nas previsões:\n")

    for k, prev in enumerate(previsoes):
        final, impacto = aplicar_correcao_simples(prev, impacto_real, k, meia_vida)

        print(f"D{k}:")
        print(f"   previsão original : {prev*100:.2f}%")
        print(f"   impacto aplicado  : {impacto*100:.2f}%")
        print(f"   previsão final    : {final*100:.2f}%\n")


if __name__ == "__main__":
    frase_teste = "aumento da demanda global por petróleo"
    run_test_frase(frase_teste)
