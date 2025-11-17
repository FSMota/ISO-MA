# ISO-MA
# Avaliação Comparativa de Algoritmos para Seleção de Genes em Dados de Microarray

Este repositório contém a implementação do Algoritmo Memético Autoral (MA) desenvolvido para o artigo:

**"Avaliação Comparativa do Isomap-GA, Hill Climbing e um Algoritmo Memético Autoral para Seleção de Genes em Dados de Microarray de Câncer no Sistema nervoso central"**

## 🚀 Equipe (Autores)

* Emmanuel Araujo Toscano Faceiro Lima
* Filipe Simões Mota
* Jose Anthony Dantas Santana
* Jose Augusto Oliveira Ferreira

*Universidade Federal de Alagoas (UFAL), Maceió, Brasil*

---

## 📄 Artigo de Base (ISO-GA)

Nossa abordagem (MA) e o Hill Climbing (HC) são comparados diretamente com a implementação do **Isomap-GA**, um algoritmo de referência proposto por Wang et al. (2020).

* **Título:** Genetic algorithm-based feature selection with manifold learning for cancer classification using microarray data
* **Autores:** Wang, Z., Zhou, Y., Takagi, T. et al.
* **Publicação:** *Scientific Reports*, vol. 10, no. 1, p. 11967, 2020.
* **DOI:** [10.1038/s41598-020-68815-0](https://doi.org/10.1038/s41598-020-68815-0)
* **Link:** [https://www.nature.com/articles/s41598-020-68815-0](https://www.nature.com/articles/s41598-020-68815-0)

---

## 📊 Log de Resultados (N=9 Execuções)

Abaixo está o log de saída completo das 9 execuções estatísticas independentes utilizadas para gerar as tabelas e boxplots do artigo. A 10ª execução foi interrompida.

```bash
Dados carregados: 60 amostras, 7129 features.
Orçamento Total de Avaliações definido para: 201000
Execuções estatísticas definidas para: 10
Iniciando 10 execuções estatísticas completas...
Isso pode levar muito tempo.

======================================================================
INICIANDO RODADA ESTATÍSTICA 1/10
======================================================================

==================================================
Iniciando Execução 1: ALGORITMO MEMÉTICO (MA)
Otimizando: 1/(1+DB) - Penalidade (FORTE)
==================================================
População inicial avaliada. Melhor Fitness Interno: 0.2390950057008993
MA Progresso: 10% | Avaliações: 20100 | Melhor Interno: 0.35729294334074874
MA Progresso: 20% | Avaliações: 40200 | Melhor Interno: 0.3771894904647392
MA Progresso: 30% | Avaliações: 60300 | Melhor Interno: 0.39238615224795237
MA Progresso: 40% | Avaliações: 80400 | Melhor Interno: 0.40328818143023915
MA Progresso: 50% | Avaliações: 100500 | Melhor Interno: 0.41139795799426637
MA Progresso: 60% | Avaliações: 120600 | Melhor Interno: 0.41870682481612287
MA Progresso: 70% | Avaliações: 140700 | Melhor Interno: 0.42291130795360377
MA Progresso: 80% | Avaliações: 160800 | Melhor Interno: 0.424150831784384
MA Progresso: 90% | Avaliações: 180900 | Melhor Interno: 0.4263733510430928
MA Progresso: 100% | Avaliações: 201000 | Melhor Interno: 0.4297679449514381
--- Fim da Execução (MA) ---

==================================================
Iniciando Execução 2: HILL CLIMBING (HC) com Reinícios
Otimizando: 1/(1+DB) - Penalidade (FORTE)
==================================================
Número total de reinícios de HC: 1005
HC Progresso: 10% | Reinício 101/1005 | Melhor Interno: 0.36580086114310467
...
(O log de execução continua até o final)
...
==================================================
Iniciando Execução 3: ALGORITMO BASE (ISO-GA)
Otimizando: DB Score Puro (Minimizar)
Rodando 10 execuções de 200 gerações...
==================================================
Iniciando execução 1/10...
Execução 1 concluída.
...
Execução 10 concluída.

Calculando solução final com Limiar Theta...
--- Fim da Execução (ISO-GA) ---
Fitness Final (DB Score) do ISO-GA: 1.841531 (Menor é melhor)

--- Avaliação Externa da Rodada 1 ---
Calculando DB Score (puro) para as soluções do MA e HC...
DB MA: 1.1118, Genes MA: 78
DB HC: 1.7324, Genes HC: 68

--- Avaliação SVM da Rodada {i+1} ---
[MA (Rodada 1)] 78 genes selecionados -> Acurácia média (5-fold): 0.7667
[HC (Rodada 1)] 68 genes selecionados -> Acurácia média (5-fold): 0.6833
[ISO-GA (Rodada 1)] 5 genes selecionados -> Acurácia média (5-fold): 0.6500

======================================================================
INICIANDO RODADA ESTATÍSTICA 2/10
======================================================================
...
(Log da Rodada 2)
...
--- Avaliação Externa da Rodada 2 ---
DB MA: 0.8651, Genes MA: 129
DB HC: 1.8019, Genes HC: 84
[MA (Rodada 2)] 129 genes selecionados -> Acurácia média (5-fold): 0.7167
[HC (Rodada 2)] 84 genes selecionados -> Acurácia média (5-fold): 0.7500
[ISO-GA (Rodada 2)] 15 genes selecionados -> Acurácia média (5-fold): 0.6833

======================================================================
INICIANDO RODADA ESTATÍSTICA 3/10
======================================================================
...
(Log da Rodada 3)
...
--- Avaliação Externa da Rodada 3 ---
DB MA: 0.9138, Genes MA: 109
DB HC: 1.7146, Genes HC: 67
[MA (Rodada 3)] 109 genes selecionados -> Acurácia média (5-fold): 0.7667
[HC (Rodada 3)] 67 genes selecionados -> Acurácia média (5-fold): 0.6667
[ISO-GA (Rodada 3)] 13 genes selecionados -> Acurácia média (5-fold): 0.6833

======================================================================
INICIANDO RODADA ESTATÍSTICA 4/10
======================================================================
...
(Log da Rodada 4)
...
--- Avaliação Externa da Rodada 4 ---
DB MA: 0.9395, Genes MA: 105
DB HC: 1.6478, Genes HC: 66
[MA (Rodada 4)] 105 genes selecionados -> Acurácia média (5-fold): 0.6833
[HC (Rodada 4)] 66 genes selecionados -> Acurácia média (5-fold): 0.6167
[ISO-GA (Rodada 4)] 12 genes selecionados -> Acurácia média (5-fold): 0.5500

======================================================================
INICIANDO RODADA ESTATÍSTICA 5/10
======================================================================
...
(Log da Rodada 5)
...
--- Avaliação Externa da Rodada 5 ---
DB MA: 1.3835, Genes MA: 42
DB HC: 1.8280, Genes HC: 67
[MA (Rodada 5)] 42 genes selecionados -> Acurácia média (5-fold): 0.7833
[HC (Rodada 5)] 67 genes selecionados -> Acurácia média (5-fold): 0.6000
[ISO-GA (Rodada 5)] 13 genes selecionados -> Acurácia média (5-fold): 0.6167

======================================================================
INICIANDO RODADA ESTATÍSTICA 6/10
======================================================================
...
(Log da Rodada 6)
...
--- Avaliação Externa da Rodada 6 ---
DB MA: 0.8869, Genes MA: 96
DB HC: 1.7770, Genes HC: 61
[MA (Rodada 6)] 96 genes selecionados -> Acurácia média (5-fold): 0.7167
[HC (Rodada 6)] 61 genes selecionados -> Acurácia média (5-fold): 0.6333
[ISO-GA (Rodada 6)] 20 genes selecionados -> Acurácia média (5-fold): 0.7167

======================================================================
INICIANDO RODADA ESTATÍSTICA 7/10
======================================================================
...
(Log da Rodada 7)
...
--- Avaliação Externa da Rodada 7 ---
DB MA: 1.0466, Genes MA: 86
DB HC: 1.5309, Genes HC: 57
[MA (Rodada 7)] 86 genes selecionados -> Acurácia média (5-fold): 0.7667
[HC (Rodada 7)] 57 genes selecionados -> Acurácia média (5-fold): 0.6000
[ISO-GA (Rodada 7)] 9 genes selecionados -> Acurácia média (5-fold): 0.7167

======================================================================
INICIANDO RODADA ESTATÍSTICA 8/10
======================================================================
...
(Log da Rodada 8)
...
--- Avaliação Externa da Rodada 8 ---
DB MA: 0.8990, Genes MA: 104
DB HC: 1.7455, Genes HC: 82
[MA (Rodada 8)] 104 genes selecionados -> Acurácia média (5-fold): 0.7167
[HC (Rodada 8)] 82 genes selecionados -> Acurácia média (5-fold): 0.6167
[ISO-GA (Rodada 8)] 5 genes selecionados -> Acurácia média (5-fold): 0.7167

======================================================================
INICIANDO RODADA ESTATÍSTICA 9/10
======================================================================
...
(Log da Rodada 9)
...
--- Avaliação Externa da Rodada 9 ---
DB MA: 0.8302, Genes MA: 130
DB HC: 1.6208, Genes HC: 52
[MA (Rodada 9)] 130 genes selecionados -> Acurácia média (5-fold): 0.7333
[HC (Rodada 9)] 52 genes selecionados -> Acurácia média (5-fold): 0.7500
[ISO-GA (Rodada 9)] 18 genes selecionados -> Acurácia média (5-fold): 0.6167

======================================================================
INICIANDO RODADA ESTATÍSTICA 10/10
======================================================================
...
(Execução interrompida)
...
```
