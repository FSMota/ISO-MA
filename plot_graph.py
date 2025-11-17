import matplotlib.pyplot as plt
import numpy as np

# --- 1. BLOCO DE DADOS (PREENCHIDO COM SEUS 9 RESULTADOS) ---
# Extraído do seu log de execução.
resultados = {
#...
}
# -----------------------------------------------------

# --- 2. FUNÇÃO DE GERAÇÃO DE BOXPLOTS ---
def gerar_graficos_boxplot(resultados):
    """
    Gera gráficos de boxplot para as métricas de resultado.
    """
    if len(resultados['db_ma']) == 0:
        print("Erro: Listas de resultados estão vazias.")
        return

    n_execucoes = len(resultados['db_ma'])
    print(f"Gerando boxplots com base em {n_execucoes} execuções...")

    # 1. Boxplot DB Score
    try:
        plt.figure(figsize=(10, 7))
        dados_db = [resultados['db_ma'], resultados['db_hc'], resultados['db_iso']]
        plt.boxplot(dados_db, labels=['MA (Proposta)', 'HC (Controle)', 'ISO-GA (Base)'],
                    patch_artist=True, medianprops={'color':'black'})
        plt.title(f'Boxplot: DB Score Final ({n_execucoes} Execuções)', fontsize=16)
        plt.ylabel('DB Score (Menor é Melhor)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('boxplot_db_score.png')
        print("Gráfico 'boxplot_db_score.png' salvo.")
    except Exception as e:
        print(f"Erro ao gerar boxplot DB Score: {e}")

    # 2. Boxplot Acurácia SVM
    try:
        plt.figure(figsize=(10, 7))
        dados_svm = [resultados['acc_ma'], resultados['acc_hc'], resultados['acc_iso']]
        plt.boxplot(dados_svm, labels=['MA (Proposta)', 'HC (Controle)', 'ISO-GA (Base)'],
                    patch_artist=True, medianprops={'color':'black'})
        plt.title(f'Boxplot: Acurácia SVM Final ({n_execucoes} Execuções)', fontsize=16)
        plt.ylabel('Acurácia SVM (Maior é Melhor)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('boxplot_svm_accuracy.png')
        print("Gráfico 'boxplot_svm_accuracy.png' salvo.")
    except Exception as e:
        print(f"Erro ao gerar boxplot SVM: {e}")

    # 3. Boxplot Nº Genes
    try:
        plt.figure(figsize=(10, 7))
        dados_genes = [resultados['genes_ma'], resultados['genes_hc'], resultados['genes_iso']]
        plt.boxplot(dados_genes, labels=['MA (Proposta)', 'HC (Controle)', 'ISO-GA (Base)'],
                    patch_artist=True, medianprops={'color':'black'})
        plt.title(f'Boxplot: Nº de Genes Selecionados ({n_execucoes} Execuções)', fontsize=16)
        plt.ylabel('Nº de Genes (Parcimônia)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('boxplot_genes.png')
        print("Gráfico 'boxplot_genes.png' salvo.")
    except Exception as e:
        print(f"Erro ao gerar boxplot Genes: {e}")

# --- 3. FUNÇÃO PARA IMPRIMIR TABELA DE MÉDIA ---
def printar_estatisticas(lista_valores, precisao_float=4):
    """
    Calcula e formata a média e o desvio padrão de uma lista de valores.
    """
    media = np.mean(lista_valores)
    std = np.std(lista_valores)
    
    if precisao_float == 0:
         return f"{media:.0f} ± {std:.0f}"
         
    return f"{media:.{precisao_float}f} ± {std:.{precisao_float}f}"

# --- 4. EXECUÇÃO PRINCIPAL (SÓ PLOTAGEM) ---
if __name__ == "__main__":
    if len(resultados['db_ma']) > 0:
        N_EXECUCOES_TOTAIS = len(resultados['db_ma'])
        print("="*70)
        print(f"COMPILAÇÃO FINAL - RESULTADOS DE {N_EXECUCOES_TOTAIS} RODADAS (MANUAL)")
        print("="*70)

        print("\n### Tabela de Resultados (Média ± Desvio Padrão) ###")
        print("--------------------------------------------------------------------------------------")
        print(f"MÉTRICA \t\t MA (Proposta) \t\t HC (Controle) \t\t ISO-GA (Base)")
        print("--------------------------------------------------------------------------------------")
        print(f"DB Score (Qualidade) \t {printar_estatisticas(resultados['db_ma']):<22} \t {printar_estatisticas(resultados['db_hc']):<22} \t {printar_estatisticas(resultados['db_iso']):<22}")
        print(f"Nº Genes (Parcimônia) \t {printar_estatisticas(resultados['genes_ma'], 0):<22} \t {printar_estatisticas(resultados['genes_hc'], 0):<22} \t {printar_estatisticas(resultados['genes_iso'], 0):<22}")
        
        print("\n### Tabela de Avaliação SVM (Média ± Desvio Padrão) ###")
        print("--------------------------------------------------------------------------------------")
        print(f"Acurácia SVM (Maior) \t {printar_estatisticas(resultados['acc_ma']):<22} \t {printar_estatisticas(resultados['acc_hc']):<22} \t {printar_estatisticas(resultados['acc_iso']):<22}")
        print("="*70)
        
        # Gerar os gráficos de boxplot
        gerar_graficos_boxplot(resultados)
        
        print("\nGráficos de boxplot gerados com sucesso.")
        print("AVISO: Os gráficos de convergência não podem ser gerados por este script,")
        print("pois os dados de progresso (a evolução interna) não foram salvos manualmente.")

    else:
        print("O dicionário 'resultados' no BLOCO 1 está vazio.")
