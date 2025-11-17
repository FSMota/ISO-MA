import pandas as pd
from scipy.io import arff
from sklearn.manifold import Isomap
from sklearn.metrics import davies_bouldin_score
import random
import os
import time
import math # Para o cálculo do orçamento
import numpy as np # Necessário para o Iso-GA
import matplotlib.pyplot as plt # Para os gráficos

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

def penalty_linear(frac, alpha=2.0):
    return alpha * frac

def penalty_quadratic(frac, alpha=2.0):
    return alpha * (frac ** 2)

def penalty_exponential(frac, alpha=5.0):
    return 1 - math.exp(-alpha * frac)

def penalty_inverse(frac, alpha=5.0):
    return frac / (alpha + frac)

def penalty_multiplicative(frac, alpha=5.0):
    # multiplicative penalty returns a multiplier, not a subtraction
    return math.exp(-alpha * frac)

def penalty_zero(_, __=None):
    return 0.0

PENALTY_TYPES = {
    "linear": penalty_linear,
    "quadratic": penalty_quadratic,
    "exp": penalty_exponential,
    "inverse": penalty_inverse,
    "mult": penalty_multiplicative,
    "none": penalty_zero
}

def compute_penalty(frac, penalty_type="quadratic", alpha=2.0):
    if penalty_type not in PENALTY_TYPES:
        raise ValueError(f"Penalty '{penalty_type}' not recognized.")

    return PENALTY_TYPES[penalty_type](frac, alpha)

# --- 1. PARÂMETROS GLOBAIS ---

# --- NOVO ---
# Define o número de execuções estatísticas. 10 é o mínimo. 30 é o ideal.
N_EXECUCOES_TOTAIS = 10
# --- FIM NOVO ---

MAX_GERACOES_ORCAMENTO = 2009
TAM_POP = 100
N_FEATURES = 7129 # Será atualizado após carregar os dados
TAM_TORNEIO = 3
TAXA_CROSSOVER = 1.0
TAXA_MUTACAO_POR_INDIVIDUO = 0.5
P_ATIVO_INICIAL = 0.005 # Inicialização Esparsa

ORCAMENTO_AVALIACOES = TAM_POP + (MAX_GERACOES_ORCAMENTO * TAM_POP)
BUDGET_POR_RESTART_HC = 200
BUDGET_BUSCA_LOCAL_MEMETICA = 50

MAX_GERACOES_ISO_GA = 200
NUM_EXECUCOES_ISO_GA = 10
LIMIAR_THETA = 0.3 # <<< CORRIGIDO: Reduzido de 0.6 para 0.3 para evitar 0 genes

CACHE_FITNESS_MA_HC = {}
CACHE_FITNESS_ISO_GA = {}

# --- 2. CARREGAMENTO DOS DADOS ---
# (Sem alterações)
try:
    data, meta = arff.loadarff('./CNS.arff')
except FileNotFoundError:
    print("ERRO: Arquivo 'CNS.arff' não encontrado. Verifique o caminho.")
    data = None
except Exception as e:
    print(f"Erro ao carregar o ARFF: {e}")
    data = None

if data is not None:
    df = pd.DataFrame(data)

    str_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in str_cols:
        df[col] = df[col].str.decode('utf-8')

    NOME_COLUNA_CLASSE = 'CLASS'
    labels_reais = None
    dados_originais = None

    if NOME_COLUNA_CLASSE not in df.columns:
        print(f"ERRO: A coluna '{NOME_COLUNA_CLASSE}' não foi encontrada.")
    else:
        labels_reais = df[NOME_COLUNA_CLASSE]
        dados_originais = df.drop(columns=[NOME_COLUNA_CLASSE])
        N_FEATURES = dados_originais.shape[1]
        print(f"Dados carregados: {dados_originais.shape[0]} amostras, {N_FEATURES} features.")
        print(f"Orçamento Total de Avaliações definido para: {ORCAMENTO_AVALIACOES}")
        print(f"Execuções estatísticas definidas para: {N_EXECUCOES_TOTAIS}")
else:
    dados_originais = None
    print("Script interrompido pois os dados não puderam ser carregados.")


# --- 3. FUNÇÕES AUXILIARES COMUNS ---
# (Sem alterações)
def criar_individuo_aleatorio(num_features, p_ativo=P_ATIVO_INICIAL):
    ind = [1 if random.random() < p_ativo else 0 for _ in range(num_features)]
    if sum(ind) == 0:
        idx_aleatorio = random.randint(0, num_features - 1)
        ind[idx_aleatorio] = 1
    return ind

def extrair_indices_ativos(individuo):
    return [i for i, bit in enumerate(individuo) if bit == 1]

def selecao_por_torneio(populacao_aferida, k):
    participantes = random.sample(populacao_aferida, k)
    vencedor = max(participantes, key=lambda item: item[1])
    return vencedor[0]

def crossover_uniforme(pai1, pai2, p_c):
    filho1, filho2 = pai1.copy(), pai2.copy()
    if random.random() >= p_c:
        return filho1, filho2
    for i in range(N_FEATURES):
        if random.random() < 0.5:
            filho1[i] = pai2[i]
            filho2[i] = pai1[i]
    return filho1, filho2

def mutacao_single_bit_flip(individuo):
    individuo_mutado = individuo.copy()
    indice_mutacao = random.randint(0, N_FEATURES - 1)
    individuo_mutado[indice_mutacao] = 1 - individuo_mutado[indice_mutacao]
    return individuo_mutado

# --- 4. FUNÇÕES DE FITNESS (OBJETIVOS INTERNOS) ---
# (Sem alterações)
def calcular_fitness_ma_hc(individuo, penalty_type="quadratic", alpha=2.0):
    global CACHE_FITNESS_MA_HC
    colunas_selecionadas = extrair_indices_ativos(individuo)
    individuo_key = frozenset(colunas_selecionadas)

    if individuo_key in CACHE_FITNESS_MA_HC:
        return CACHE_FITNESS_MA_HC[individuo_key]

    if len(colunas_selecionadas) == 0: return 0.0
    dados_reduzidos = dados_originais.iloc[:, colunas_selecionadas].to_numpy()

    frac = len(colunas_selecionadas) / N_FEATURES
    penalty = compute_penalty(frac, penalty_type, alpha)

    try:
        dados_projetados = Isomap(n_components=2, n_neighbors=10).fit_transform(dados_reduzidos)
        score_db = davies_bouldin_score(dados_projetados, labels_reais)
        fitness = 1.0 / (score_db + 1.0) - penalty
    except Exception as e:
        fitness = 0.0

    CACHE_FITNESS_MA_HC[individuo_key] = fitness
    return fitness

def calcular_fitness_iso_ga(individuo):
    global CACHE_FITNESS_ISO_GA
    colunas_selecionadas = extrair_indices_ativos(individuo)
    individuo_key = frozenset(colunas_selecionadas)

    if individuo_key in CACHE_FITNESS_ISO_GA:
        return CACHE_FITNESS_ISO_GA[individuo_key]

    if len(colunas_selecionadas) == 0: return float('inf')
    dados_reduzidos = dados_originais.iloc[:, colunas_selecionadas].to_numpy()

    try:
        dados_projetados = Isomap(n_components=2, n_neighbors=10).fit_transform(dados_reduzidos)
        score_db = davies_bouldin_score(dados_projetados, labels_reais)
    except Exception as e:
        score_db = float('inf')

    CACHE_FITNESS_ISO_GA[individuo_key] = score_db
    return score_db


# --- 5. FUNÇÕES DE BUSCA LOCAL (PARA MA e HC) ---
# (Sem alterações)
def busca_local_hill_climbing(individuo_bruto, B, penalty_type="quadratic", alpha=2.0):
    individuo_atual = individuo_bruto.copy()
    fitness_atual = calcular_fitness_ma_hc(individuo_atual, penalty_type, alpha)
    avaliacoes_gastas = 1

    for _ in range(B - 1):
        vizinho = individuo_atual.copy()
        indice_flip = random.randint(0, N_FEATURES - 1)
        vizinho[indice_flip] = 1 - vizinho[indice_flip]
        fitness_vizinho = calcular_fitness_ma_hc(vizinho, penalty_type, alpha)
        avaliacoes_gastas += 1

        if fitness_vizinho > fitness_atual:
            individuo_atual = vizinho
            fitness_atual = fitness_vizinho

    return individuo_atual, fitness_atual, avaliacoes_gastas


# --- 6. ALGORITMO 1: ALGORITMO MEMÉTICO (MA) ---
# (Sem alterações)
def executar_algoritmo_memetico(penalty_type="quadratic", alpha=2.0):
    print("\n" + "="*50)
    print("Iniciando Execução 1: ALGORITMO MEMÉTICO (MA)")
    print(f"Otimizando: 1/(1+DB) - Penalidade (FORTE)")
    print("="*50)
    start_time = time.time()
    global CACHE_FITNESS_MA_HC
    CACHE_FITNESS_MA_HC = {}
    num_avaliacoes = 0
    População_Aferida = []
    progresso_ma = []

    for _ in range(TAM_POP):
        ind_aleatorio = criar_individuo_aleatorio(N_FEATURES)
        fitness = calcular_fitness_ma_hc(ind_aleatorio, penalty_type, alpha)
        num_avaliacoes += 1
        População_Aferida.append((ind_aleatorio, fitness))

    Melhor_Global = max(População_Aferida, key=lambda item: item[1])
    progresso_ma.append((num_avaliacoes, Melhor_Global[1]))
    print(f"População inicial avaliada. Melhor Fitness Interno: {Melhor_Global[1]}")

    while num_avaliacoes < ORCAMENTO_AVALIACOES:
        pai1 = selecao_por_torneio(População_Aferida, TAM_TORNEIO)
        pai2 = selecao_por_torneio(População_Aferida, TAM_TORNEIO)
        filho1, filho2 = crossover_uniforme(pai1, pai2, TAXA_CROSSOVER)
        filho_base = random.choice([filho1, filho2])

        if random.random() < TAXA_MUTACAO_POR_INDIVIDUO:
            filho_final = mutacao_single_bit_flip(filho_base)
        else:
            filho_final = filho_base

        budget_disponivel = ORCAMENTO_AVALIACOES - num_avaliacoes
        budget_para_polir = min(BUDGET_BUSCA_LOCAL_MEMETICA, budget_disponivel)
        if budget_para_polir <= 1: break

        (filho_polido, fit_filho, evals_gastas_hc) = busca_local_hill_climbing(
            filho_final, budget_para_polir, penalty_type, alpha
        )
        num_avaliacoes += evals_gastas_hc

        pior_idx = min(range(len(População_Aferida)), key=lambda i: População_Aferida[i][1])
        if fit_filho > População_Aferida[pior_idx][1]:
            População_Aferida[pior_idx] = (filho_polido, fit_filho)
            if fit_filho > Melhor_Global[1]:
                Melhor_Global = (filho_polido, fit_filho)
                progresso_ma.append((num_avaliacoes, Melhor_Global[1]))

        if num_avaliacoes % (ORCAMENTO_AVALIACOES // 10) < BUDGET_BUSCA_LOCAL_MEMETICA:
            progresso = (num_avaliacoes / ORCAMENTO_AVALIACOES) * 100
            print(f"MA Progresso: {progresso:.0f}% | Avaliações: {num_avaliacoes} | Melhor Interno: {Melhor_Global[1]}")

    end_time = time.time()
    tempo_total = end_time - start_time
    print("--- Fim da Execução (MA) ---")
    return Melhor_Global[0], num_avaliacoes, tempo_total, progresso_ma


# --- 7. ALGORITMO 2: HILL CLIMBING (HC) ---
# (Sem alterações)
def executar_hill_climbing_restarts(penalty_type="quadratic", alpha=2.0):
    print("\n" + "="*50)
    print("Iniciando Execução 2: HILL CLIMBING (HC) com Reinícios")
    print(f"Otimizando: 1/(1+DB) - Penalidade (FORTE)")
    print("="*50)
    start_time = time.time()
    global CACHE_FITNESS_MA_HC
    CACHE_FITNESS_MA_HC = {}
    num_restarts = ORCAMENTO_AVALIACOES // BUDGET_POR_RESTART_HC
    print(f"Número total de reinícios de HC: {num_restarts}")

    melhor_global_hc = (None, 0.0)
    num_avaliacoes = 0
    progresso_hc = []
    progresso_hc.append((0, 0.0))

    for i in range(num_restarts):
        ind_aleatorio = criar_individuo_aleatorio(N_FEATURES)
        (ind_local, fit_local, evals_gastas) = busca_local_hill_climbing(
            ind_aleatorio, BUDGET_POR_RESTART_HC, penalty_type, alpha
        )
        num_avaliacoes += evals_gastas

        if fit_local > melhor_global_hc[1]:
            melhor_global_hc = (ind_local, fit_local)
            progresso_hc.append((num_avaliacoes, melhor_global_hc[1]))

        if (i+1) % (num_restarts // 10 + 1) == 0:
            progresso = (i+1) / num_restarts * 100
            print(f"HC Progresso: {progresso:.0f}% | Reinício {i+1}/{num_restarts} | Melhor Interno: {melhor_global_hc[1]}")

    end_time = time.time()
    tempo_total = end_time - start_time
    print("--- Fim da Execução (HC) ---")
    return melhor_global_hc[0], num_avaliacoes, tempo_total, progresso_hc

# --- 8. ALGORITMO 3: ISO-GA (O BASE) ---
# (Sem alterações)
def selecao_por_torneio_min(populacao_aferida, k):
    participantes = random.sample(populacao_aferida, k)
    vencedor = min(participantes, key=lambda item: item[1])
    return vencedor[0]

def executar_uma_rodada_iso_ga():
    global CACHE_FITNESS_ISO_GA
    CACHE_FITNESS_ISO_GA = {}
    progresso_run = []

    populacao = [criar_individuo_aleatorio(N_FEATURES) for _ in range(TAM_POP)]
    pop_aferida = [(ind, calcular_fitness_iso_ga(ind)) for ind in populacao]
    avaliacoes_nesta_rodada = TAM_POP
    melhor_da_rodada = min(pop_aferida, key=lambda item: item[1])
    progresso_run.append((avaliacoes_nesta_rodada, melhor_da_rodada[1]))

    for ger in range(MAX_GERACOES_ISO_GA):
        nova_populacao = []
        (melhor_ind_ger, melhor_fit_ger) = min(pop_aferida, key=lambda item: item[1])
        nova_populacao.append(melhor_ind_ger)

        while len(nova_populacao) < TAM_POP:
            pai1 = selecao_por_torneio_min(pop_aferida, TAM_TORNEIO)
            pai2 = selecao_por_torneio_min(pop_aferida, TAM_TORNEIO)
            filho1, filho2 = crossover_uniforme(pai1, pai2, TAXA_CROSSOVER)
            filho = random.choice([filho1, filho2])
            if random.random() < TAXA_MUTACAO_POR_INDIVIDUO:
                filho = mutacao_single_bit_flip(filho)
            nova_populacao.append(filho)

        populacao = nova_populacao
        pop_aferida = [(ind, calcular_fitness_iso_ga(ind)) for ind in populacao]
        avaliacoes_nesta_rodada += len(pop_aferida)

        melhor_da_geracao_atual = min(pop_aferida, key=lambda item: item[1])
        if melhor_da_geracao_atual[1] < melhor_da_rodada[1]:
            melhor_da_rodada = melhor_da_geracao_atual
            progresso_run.append((avaliacoes_nesta_rodada, melhor_da_rodada[1]))

    return melhor_da_rodada[0], progresso_run

def executar_iso_ga_completo():
    print("\n" + "="*50)
    print("Iniciando Execução 3: ALGORITMO BASE (ISO-GA)")
    print(f"Otimizando: DB Score Puro (Minimizar)")
    print(f"Rodando {NUM_EXECUCOES_ISO_GA} execuções de {MAX_GERACOES_ISO_GA} gerações...")
    print("="*50)
    start_time = time.time()
    lista_de_melhores_individuos = []
    lista_de_progressos_iso_ga = []

    for i in range(NUM_EXECUCOES_ISO_GA):
        print(f"Iniciando execução {i+1}/{NUM_EXECUCOES_ISO_GA}...")
        melhor_ind_da_rodada, progresso_da_rodada = executar_uma_rodada_iso_ga()
        lista_de_melhores_individuos.append(melhor_ind_da_rodada)
        lista_de_progressos_iso_ga.append(progresso_da_rodada)
        print(f"Execução {i+1} concluída.")

    print("\nCalculando solução final com Limiar Theta...")
    soma_dos_vetores = np.sum(lista_de_melhores_individuos, axis=0)
    limite_contagem = int(NUM_EXECUCOES_ISO_GA * LIMIAR_THETA)
    solucao_final_iso_ga = [
        1 if contagem >= limite_contagem else 0
        for contagem in soma_dos_vetores
    ]
    end_time = time.time()
    tempo_total = end_time - start_time
    print("--- Fim da Execução (ISO-GA) ---")

    fitness_final = calcular_fitness_iso_ga(solucao_final_iso_ga)
    print(f"Fitness Final (DB Score) do ISO-GA: {fitness_final:.6f} (Menor é melhor)")

    return solucao_final_iso_ga, fitness_final, tempo_total, lista_de_progressos_iso_ga


# --- 9. BLOCO DE GERAÇÃO DE GRÁFICOS ---

# --- MODIFICADO ---
# Renomeado para 'convergencia'
def gerar_graficos_convergencia(prog_ma, prog_hc, prog_iso_ga):
    print("\nGerando gráficos de convergência da rodada mediana...")

    # --- Gráfico 1: MA vs HC (Otimizando Fitness Interno - Maximizando) ---
    try:
        plt.figure(figsize=(12, 8))
        ma_x = [p[0] for p in prog_ma]
        ma_y = [p[1] for p in prog_ma]
        plt.plot(ma_x, ma_y, label="Algoritmo Memético (MA)", marker='o', markersize=3, linestyle='-')

        hc_x = [p[0] for p in prog_hc]
        hc_y = [p[1] for p in prog_hc]
        plt.plot(hc_x, hc_y, label="Hill Climbing (HC)", marker='x', markersize=3, linestyle='--')

        plt.title('Convergência Interna (Rodada Mediana): MA vs HC (Orçamento 201.000)')
        plt.xlabel('Avaliações de Fitness')
        plt.ylabel('Melhor Fitness Interno (Maior é Melhor)')
        plt.legend()
        plt.grid(True)
        plt.savefig('convergencia_interna_ma_hc.png')
        print("Gráfico 'convergencia_interna_ma_hc.png' salvo.")
    except Exception as e:
        print(f"Erro ao gerar gráfico MA vs HC: {e}")

    # --- Gráfico 2: ISO-GA (Otimizando Fitness Interno - Minimizando) ---
    try:
        plt.figure(figsize=(12, 8))
        for i, run_progress in enumerate(prog_iso_ga):
            run_x = [p[0] for p in run_progress]
            run_y = [p[1] for p in run_progress]
            plt.plot(run_x, run_y, label=f'Rodada {i+1}' if i < 5 else '_nolegend_', alpha=0.6)

        plt.title('Convergência Interna (Rodada Mediana): 10 Rodadas do ISO-GA (Orçamento 20.100 cada)')
        plt.xlabel('Avaliações de Fitness (por rodada)')
        plt.ylabel('Melhor Fitness Interno (Menor é Melhor - DB Score)')
        plt.legend()
        plt.grid(True)
        plt.savefig('convergencia_interna_iso_ga.png')
        print("Gráfico 'convergencia_interna_iso_ga.png' salvo.")
    except Exception as e:
        print(f"Erro ao gerar gráfico ISO-GA: {e}")

# --- NOVO ---
# Nova função para gerar os boxplots
def gerar_graficos_boxplot(resultados):
    """
    Gera gráficos de boxplot para as métricas de resultado (DB, SVM, Genes).
    """
    print("\nGerando gráficos de boxplot...")

    # 1. Boxplot DB Score
    try:
        plt.figure(figsize=(10, 7))
        dados_db = [resultados['db_ma'], resultados['db_hc'], resultados['db_iso']]
        plt.boxplot(dados_db, labels=['MA (Proposta)', 'HC (Controle)', 'ISO-GA (Base)'],
                    patch_artist=True, medianprops={'color':'black'})
        plt.title(f'Boxplot: DB Score Final ({N_EXECUCOES_TOTAIS} Execuções)', fontsize=16)
        plt.ylabel('DB Score (Menor é Melhor)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('boxplot_db_score.png')
        print("Gráfico 'boxplot_db_score.png' salvo.")
    except Exception as e:
        print(f"Erro ao gerar boxplot DB Score: {e}")

    # 2. Boxplot Acurácia SVM
    try:
        plt.figure(figsize=(10, 7))
        # Filtra valores None/0.0 que podem ter ocorrido por erro no SVM
        dados_svm_ma = [v for v in resultados['acc_ma'] if v]
        dados_svm_hc = [v for v in resultados['acc_hc'] if v]
        dados_svm_iso = [v for v in resultados['acc_iso'] if v]
        dados_svm = [dados_svm_ma, dados_svm_hc, dados_svm_iso]

        plt.boxplot(dados_svm, labels=['MA (Proposta)', 'HC (Controle)', 'ISO-GA (Base)'],
                    patch_artist=True, medianprops={'color':'black'})
        plt.title(f'Boxplot: Acurácia SVM Final ({N_EXECUCOES_TOTAIS} Execuções)', fontsize=16)
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
        plt.title(f'Boxplot: Nº de Genes Selecionados ({N_EXECUCOES_TOTAIS} Execuções)', fontsize=16)
        plt.ylabel('Nº de Genes (Parcimônia)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('boxplot_genes.png')
        print("Gráfico 'boxplot_genes.png' salvo.")
    except Exception as e:
        print(f"Erro ao gerar boxplot Genes: {e}")

# --- 10. FUNÇÃO DE AVALIAÇÃO SVM (Classificação) ---
# (Sem alterações)
def avaliar_svm(indices_genes, nome_metodo):
    if not indices_genes or sum(indices_genes) == 0:
        print(f"[{nome_metodo}] Nenhum gene selecionado, pulando...")
        return None

    idx = np.where(np.array(indices_genes) == 1)[0]
    if len(idx) == 0:
        print(f"[{nome_metodo}] Nenhum gene selecionado (após np.where), pulando...")
        return None

    X_sel = dados_originais.iloc[:, idx].to_numpy()
    y = labels_reais.to_numpy()

    try:
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale"))
        scores = cross_val_score(clf, X_sel, y, cv=5)
        acc = scores.mean()
        print(f"[{nome_metodo}] {len(idx)} genes selecionados -> Acurácia média (5-fold): {acc:.4f}")
        return acc
    except ValueError as e:
        print(f"[{nome_metodo}] Erro ao avaliar SVM (ex: 1 única classe em um fold): {e}")
        return None
    except Exception as e:
        print(f"[{nome_metodo}] Erro inesperado no SVM: {e}")
        return None

# --- 11. BLOCO DE EXECUÇÃO PRINCIPAL ---

# --- NOVO ---
# Função auxiliar para formatar Média ± Desvio Padrão
def printar_estatisticas(lista_valores, precisao_float=4):
    """
    Calcula e formata a média e o desvio padrão de uma lista de valores.
    """
    media = np.mean(lista_valores)
    std = np.std(lista_valores)
    
    # Se a precisão for 0 (para Nª de Genes), formata como inteiro
    if precisao_float == 0:
         return f"{media:.0f} ± {std:.0f}"
         
    return f"{media:.{precisao_float}f} ± {std:.{precisao_float}f}"

# --- MODIFICADO ---
# O bloco __main__ agora executa o loop principal e coleta estatísticas
if __name__ == "__main__" and dados_originais is not None:

    # Dicionário para armazenar os resultados de todas as execuções
    resultados = {
        'db_ma': [], 'genes_ma': [], 'time_ma': [], 'acc_ma': [],
        'db_hc': [], 'genes_hc': [], 'time_hc': [], 'acc_hc': [],
        'db_iso': [], 'genes_iso': [], 'time_iso': [], 'acc_iso': []
    }
    
    # Listas para armazenar os dados de progresso de CADA rodada
    todos_progressos_ma = []
    todos_progressos_hc = []
    todos_progressos_iso = []

    print(f"Iniciando {N_EXECUCOES_TOTAIS} execuções estatísticas completas...")
    print("Isso pode levar muito tempo.")

    # --- LOOP DE EXECUÇÃO ESTATÍSTICA ---
    for i in range(N_EXECUCOES_TOTAIS):
        print("\n" + "="*70)
        print(f"INICIANDO RODADA ESTATÍSTICA {i+1}/{N_EXECUCOES_TOTAIS}")
        print("="*70)

        # --- Execução 1: Algoritmo Memético ---
        (melhor_ind_ma, evals_ma, time_ma, prog_ma) = executar_algoritmo_memetico("linear", 4)
        resultados['time_ma'].append(time_ma)
        todos_progressos_ma.append(prog_ma)

        # --- Execução 2: Hill Climbing ---
        (melhor_ind_hc, evals_hc, time_hc, prog_hc) = executar_hill_climbing_restarts()
        resultados['time_hc'].append(time_hc)
        todos_progressos_hc.append(prog_hc)

        # --- Execução 3: Iso-GA ---
        (solucao_iso_ga, fit_iso_ga, time_iso_ga, prog_iso_ga) = executar_iso_ga_completo()
        resultados['time_iso'].append(time_iso_ga)
        todos_progressos_iso.append(prog_iso_ga)
        # Salva o DB e Genes do ISO-GA (já são finais)
        resultados['db_iso'].append(fit_iso_ga)
        resultados['genes_iso'].append(sum(solucao_iso_ga) if solucao_iso_ga else 0)

        # --- Avaliação Externa da Rodada (DB e Genes) ---
        print(f"\n--- Avaliação Externa da Rodada {i+1} ---")
        print("Calculando DB Score (puro) para as soluções do MA e HC...")
        db_score_ma = calcular_fitness_iso_ga(melhor_ind_ma)
        db_score_hc = calcular_fitness_iso_ga(melhor_ind_hc)
        resultados['db_ma'].append(db_score_ma)
        resultados['db_hc'].append(db_score_hc)

        genes_ma = sum(melhor_ind_ma) if melhor_ind_ma else 0
        genes_hc = sum(melhor_ind_hc) if melhor_ind_hc else 0
        resultados['genes_ma'].append(genes_ma)
        resultados['genes_hc'].append(genes_hc)
        
        print(f"DB MA: {db_score_ma:.4f}, Genes MA: {genes_ma}")
        print(f"DB HC: {db_score_hc:.4f}, Genes HC: {genes_hc}")

        # --- Avaliação SVM da Rodada ---
        print("\n--- Avaliação SVM da Rodada {i+1} ---")
        acc_ma = avaliar_svm(melhor_ind_ma, f"MA (Rodada {i+1})")
        acc_hc = avaliar_svm(melhor_ind_hc, f"HC (Rodada {i+1})")
        acc_iso = avaliar_svm(solucao_iso_ga, f"ISO-GA (Rodada {i+1})")
        
        # Armazena 0.0 se a avaliação falhar (para manter o N de execuções)
        resultados['acc_ma'].append(acc_ma if acc_ma else 0.0)
        resultados['acc_hc'].append(acc_hc if acc_hc else 0.0)
        resultados['acc_iso'].append(acc_iso if acc_iso else 0.0)

    # --- FIM DO LOOP ---

    # --- COMPILAÇÃO FINAL (ESTATÍSTICAS) ---
    print("\n" + "="*70)
    print(f"COMPILAÇÃO FINAL - RESULTADOS DE {N_EXECUCOES_TOTAIS} RODADAS")
    print("="*70)

    print("\n### Tabela de Resultados (Média ± Desvio Padrão) ###")
    print("--------------------------------------------------------------------------------------")
    print(f"MÉTRICA \t\t MA (Proposta) \t\t HC (Controle) \t\t ISO-GA (Base)")
    print("--------------------------------------------------------------------------------------")
    # Adiciona espaçamento (:<22) para alinhar as colunas
    print(f"DB Score (Qualidade) \t {printar_estatisticas(resultados['db_ma']):<22} \t {printar_estatisticas(resultados['db_hc']):<22} \t {printar_estatisticas(resultados['db_iso']):<22}")
    print(f"Nº Genes (Parcimônia) \t {printar_estatisticas(resultados['genes_ma'], 0):<22} \t {printar_estatisticas(resultados['genes_hc'], 0):<22} \t {printar_estatisticas(resultados['genes_iso'], 0):<22}")
    print(f"Tempo (s) (Custo) \t {printar_estatisticas(resultados['time_ma'], 2):<22} \t {printar_estatisticas(resultados['time_hc'], 2):<22} \t {printar_estatisticas(resultados['time_iso'], 2):<22}")
    print("--------------------------------------------------------------------------------------")

    print("\n### Tabela de Avaliação SVM (Média ± Desvio Padrão) ###")
    print("--------------------------------------------------------------------------------------")
    print(f"MÉTRICA \t\t MA (Proposta) \t\t HC (Controle) \t\t ISO-GA (Base)")
    print("--------------------------------------------------------------------------------------")
    print(f"Acurácia SVM (Maior) \t {printar_estatisticas(resultados['acc_ma']):<22} \t {printar_estatisticas(resultados['acc_hc']):<22} \t {printar_estatisticas(resultados['acc_iso']):<22}")
    print("="*70)

    # --- GERAÇÃO DE GRÁFICOS ---
    
    # Seleciona a rodada mediana (em termos de DB Score do MA) para plotar
    # Isso evita plotar uma rodada de "sorte" ou "azar"
    try:
        median_run_index = np.argsort(resultados['db_ma'])[len(resultados['db_ma']) // 2]
        print(f"\nRodada mediana selecionada para gráficos de convergência: {median_run_index+1}")
        
        # Gera os gráficos de convergência daquela rodada
        gerar_graficos_convergencia(
            todos_progressos_ma[median_run_index],
            todos_progressos_hc[median_run_index],
            todos_progressos_iso[median_run_index]
        )
    except Exception as e:
        print(f"Erro ao selecionar rodada mediana ou gerar gráficos de convergência: {e}")

    # Gera os boxplots com os dados de TODAS as rodadas
    gerar_graficos_boxplot(resultados)


elif dados_originais is None:
