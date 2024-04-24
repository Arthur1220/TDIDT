import numpy as np
import pandas as pd

class Node:
    def __init__(self, attribute, value=None):
        self.attribute = attribute
        self.value = value
        self.children = {}

def calcular_entropia(y):
    _, counts = np.unique(y, return_counts=True)
    probabilidade = counts / len(y)
    entropia = -np.sum(probabilidade * np.log2(probabilidade))
    return entropia

def calcular_ganho_informacao(dados, atributo, atributo_alvo):
    entropia_inicial = calcular_entropia(dados[atributo_alvo])

    valores_atributo = dados[atributo].unique()
    entropia_total = 0

    for valor in valores_atributo:
        subconjunto = dados[dados[atributo] == valor]
        entropia_subconjunto = calcular_entropia(subconjunto[atributo_alvo])
        proporcao = len(subconjunto) / len(dados)
        entropia_total += proporcao * entropia_subconjunto

    ganho_informacao = entropia_inicial - entropia_total
    return ganho_informacao

def criar_arvore(dados, atributos, atributo_alvo, valor=None):
    if len(np.unique(dados[atributo_alvo])) == 1:
        # Todos os exemplos têm o mesmo rótulo, retorna um nó folha
        return Node(dados[atributo_alvo].iloc[0], value=valor)

    if len(atributos) == 0:
        # Não há mais atributos para dividir, retorna o valor mais comum do atributo alvo
        valores, contagens = np.unique(dados[atributo_alvo], return_counts=True)
        indice_mais_comum = np.argmax(contagens)
        return Node(valores[indice_mais_comum], value=valor)

    # Calcula o ganho de informação para cada atributo
    ganhos_informacao = [calcular_ganho_informacao(dados, atributo, atributo_alvo) for atributo in atributos]
    indice_melhor_atributo = np.argmax(ganhos_informacao)
    melhor_atributo = atributos[indice_melhor_atributo]

    # Seleciona o atributo com o maior ganho
    arvore = Node(melhor_atributo, value=valor)

    valores_atributo = dados[melhor_atributo].unique()
    novos_atributos = [a for a in atributos if a != melhor_atributo]

    # Cria um subconjunto dos dados
    for valor in valores_atributo:
        subconjunto = dados[dados[melhor_atributo] == valor]
        if len(subconjunto) == 0:
            # Não há exemplos com esse valor do atributo, retorna o valor mais comum do atributo alvo
            valores, contagens = np.unique(dados[atributo_alvo], return_counts=True)
            indice_mais_comum = np.argmax(contagens)
            arvore.children[valor] = Node(valores[indice_mais_comum], value=valor)
        else:
            arvore.children[valor] = criar_arvore(subconjunto, novos_atributos, atributo_alvo, valor=valor)

    return arvore

def formatar_arvore(arvore, nivel=0):
    if not arvore.children:
        return arvore.attribute
    else:
        espacos = "  " * nivel
        output = ""
        for child_value, child_node in arvore.children.items():
            output += f"\n{espacos}{arvore.attribute} = {child_value}: {formatar_arvore(child_node, nivel + 1)}"
        return output

def imprimir_arvore(dados, rotulo, arvore, nome_arquivo):
    output = ""

    output += "\n\n"
    output += str(dados)
    output += "\n\nVariável final: " + str(rotulo)
    output += "\n\nÁrvore de Indução:\n"
    arvore_formatada = formatar_arvore(arvore)
    output += arvore_formatada
    output += "\n\n"

    with open(nome_arquivo, 'w') as arquivo:
        arquivo.write(output)

    print(f"A árvore foi salva no arquivo '{nome_arquivo}'")

def criar_arvore_inducao(nome_arquivo_csv, nome_arquivo_saida):
    dados = pd.read_csv(nome_arquivo_csv)
    rotulo = dados.columns[-1]
    atributos = dados.columns[:-1]

    dados[rotulo] = dados[rotulo].astype(str)  # Converter valores do atributo alvo para string

    arvore = criar_arvore(dados, atributos, rotulo)

    imprimir_arvore(dados, rotulo, arvore, nome_arquivo_saida)

# Caso 01 - Tenis
nome_arquivo_csv = 'weather.nominal.csv'
nome_arquivo_saida = 'arvore - tenis.txt'
criar_arvore_inducao(nome_arquivo_csv, nome_arquivo_saida)

# Caso 02 - Anamnese
nome_arquivo_csv = 'anamnese.csv'
nome_arquivo_saida = 'arvore - anamnese.txt'
criar_arvore_inducao(nome_arquivo_csv, nome_arquivo_saida)

# Caso 03 - Vinhos
nome_arquivo_csv = 'tabelavinho.csv'
nome_arquivo_saida = 'arvore - vinho.txt'
criar_arvore_inducao(nome_arquivo_csv, nome_arquivo_saida)