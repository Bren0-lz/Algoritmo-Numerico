import numpy as np

# --- MÉTODOS ANTERIORES ---

def eliminacao_gauss_simples(matriz_aumentada):
    """
    Realiza a eliminação de Gauss sem troca de linhas.
    Transforma a matriz estendida em uma matriz triangular superior.
    """
    n = len(matriz_aumentada)
    
    # 1. FASE DE ELIMINAÇÃO (Escalonamento)
    for k in range(n - 1): # Percorre as colunas diagonais (pivôs)
        # Verifica se o pivô é zero (o método simples falha se isso ocorrer)
        if matriz_aumentada[k, k] == 0:
            print("\nERRO: Pivô zero encontrado...")
            return None
        
        # Percorre as linhas abaixo do pivô atual para zerar os elementos
        for i in range(k + 1, n):
            # Calcula o fator m = elemento_abaixo / pivo
            multiplicador = matriz_aumentada[i, k] / matriz_aumentada[k, k]
            
            # Atualiza a linha inteira: Linha_i = Linha_i - (m * Linha_k)
            matriz_aumentada[i, :] = matriz_aumentada[i, :] - \
                multiplicador * matriz_aumentada[k, :]
                
    print("\nMatriz após a fase de eliminação (triangular superior):")
    print(np.round(matriz_aumentada, 4))
    
    # 2. FASE DE SUBSTITUIÇÃO RETROATIVA (Back Substitution)
    x = np.zeros(n) # Vetor que guardará as soluções (x1, x2, ...)
    
    # Começa da última equação (n-1) até a primeira (0), andando para trás (-1)
    for i in range(n - 1, -1, -1):
        # Soma os termos já conhecidos da equação: a_ij * x_j
        soma = np.dot(matriz_aumentada[i, i+1:n], x[i+1:n])
        
        # Isola o x_i: x_i = (b_i - soma) / a_ii
        x[i] = (matriz_aumentada[i, n] - soma) / matriz_aumentada[i, i]
        
    return x


def gauss_com_pivoteamento(matriz_aumentada):
    """
    Melhoria do método anterior. Antes de zerar a coluna, troca linhas 
    para colocar o maior valor absoluto possível na posição do pivô.
    Isso reduz erros de arredondamento e evita divisão por zero.
    """
    n = len(matriz_aumentada)
    
    for k in range(n - 1):
        # --- BUSCA DO MAIOR PIVÔ (Pivotação Parcial) ---
        max_index = k
        for i in range(k + 1, n):
            # Procura o maior valor absoluto na coluna k (abaixo da diagonal)
            if abs(matriz_aumentada[i, k]) > abs(matriz_aumentada[max_index, k]):
                max_index = i
        
        # Se achou um pivô melhor em outra linha, troca as linhas
        if max_index != k:
            print(f"\nPivotando: Trocando linha {k+1} com linha {max_index+1}")
            # Troca usando "fancy indexing" do Numpy
            matriz_aumentada[[k, max_index]] = matriz_aumentada[[max_index, k]]
            
        # Se mesmo após a troca o pivô for zero, a matriz não tem solução única
        if matriz_aumentada[k, k] == 0:
            print("\nERRO: A matriz é singular...")
            return None
        
        # --- ELIMINAÇÃO (Igual ao Gauss Simples) ---
        for i in range(k + 1, n):
            multiplicador = matriz_aumentada[i, k] / matriz_aumentada[k, k]
            matriz_aumentada[i, :] = matriz_aumentada[i, :] - \
                multiplicador * matriz_aumentada[k, :]
                
    print("\nMatriz após a fase de eliminação com pivoteamento:")
    print(np.round(matriz_aumentada, 4))
    
    # --- SUBSTITUIÇÃO RETROATIVA ---
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(matriz_aumentada[i, i+1:n], x[i+1:n])
        x[i] = (matriz_aumentada[i, n] - soma) / matriz_aumentada[i, i]
        
    return x


def decomposicao_lu(A, b):
    """
    Fatora a matriz A em L (Lower/Inferior) e U (Upper/Superior).
    Resolve o sistema em dois passos: Ly = b e depois Ux = y.
    """
    n = len(A)
    L = np.eye(n) # L começa como matriz identidade (1 na diagonal)
    U = A.copy()  # U começa como cópia de A e será escalonada
    
    # 1. FATORAÇÃO LU
    for k in range(n - 1):
        for i in range(k + 1, n):
            if U[k, k] == 0:
                print("\nERRO: Pivô zero encontrado. A decomposição LU simples falhou.")
                return None
            
            # Calcula o multiplicador e guarda na matriz L
            multiplicador = U[i, k] / U[k, k]
            L[i, k] = multiplicador 
            
            # Aplica a eliminação na matriz U
            U[i, :] = U[i, :] - multiplicador * U[k, :]
            
    print("\nMatriz L (Triangular Inferior):")
    print(np.round(L, 4))
    print("\nMatriz U (Triangular Superior):")
    print(np.round(U, 4))
    
    # 2. SUBSTITUIÇÃO PROGRESSIVA (Resolve Ly = b)
    # Descobre o vetor intermediário y
    y = np.zeros(n)
    for i in range(n):
        # Soma dos elementos anteriores na linha de L
        soma = np.dot(L[i, :i], y[:i])
        y[i] = b[i] - soma
        
    # 3. SUBSTITUIÇÃO RETROATIVA (Resolve Ux = y)
    # Descobre o vetor final x
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (y[i] - soma) / U[i, i]
        
    return x


# --- NOVO MÉTODO: DECOMPOSIÇÃO LUP ---

def decomposicao_lup(A, b):
    """
    Resolve um sistema Ax=b usando Decomposição LUP (com Pivotação Parcial).
    Decompõe PA = LU, onde P é uma matriz de permutação.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = A.copy().astype(float)
    P = np.eye(n)  # Matriz de permutação começa como identidade

    for k in range(n - 1):
        # --- ETAPA DE PIVOTAÇÃO ---
        # Encontra o índice do maior valor na coluna k (a partir da linha k)
        max_index = k + np.argmax(np.abs(U[k:, k]))
        
        if max_index != k:
            print(f"\nPivotando: Trocando linha {k+1} com linha {max_index+1}")
            # Troca as linhas em U (parte superior)
            U[[k, max_index]] = U[[max_index, k]]
            # Troca as linhas em P (registra a permutação)
            P[[k, max_index]] = P[[max_index, k]]
            # Troca as linhas em L (parte já calculada da triangular inferior)
            # Isso é crucial para manter a consistência matemática
            L[[k, max_index]] = L[[max_index, k]]

        # --- FIM DA PIVOTAÇÃO ---

        if U[k, k] == 0:
            print("\nERRO: A matriz é singular.")
            return None

        # O resto da decomposição (Cálculo dos multiplicadores)
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k] # Guarda o multiplicador em L
            U[i, :] -= L[i, k] * U[k, :] # Zera o elemento em U

    np.fill_diagonal(L, 1)  # Preenche a diagonal de L com 1s (Convenção de Doolittle)

    print("\nMatriz P (Permutação):")
    print(np.round(P, 4))
    print("\nMatriz L (Triangular Inferior):")
    print(np.round(L, 4))
    print("\nMatriz U (Triangular Superior):")
    print(np.round(U, 4))

    # --- FASE 2: RESOLUÇÃO DO SISTEMA ---
    # Como PAx = Pb e PA = LU, então LUx = Pb.
    
    # Passo A: Aplicar as permutações em b (Gerar vetor Pb)
    b_permutado = np.dot(P, b)

    # Passo B: Substituição Progressiva (Ly = b_permutado)
    y = np.zeros(n)
    for i in range(n):
        soma = np.dot(L[i, :i], y[:i])
        y[i] = b_permutado[i] - soma

    # Passo C: Substituição Retroativa (Ux = y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (y[i] - soma) / U[i, i]

    return x

# --- FUNÇÕES DE INTERFACE (com menu atualizado) ---


def obter_dados_do_usuario():
    # ... (código inalterado)
    print("--------------------------------------------------")
    print("  Resolvedor de Sistemas de Equações Lineares    ")
    print("--------------------------------------------------")
    while True:
        try:
            # Solicita tamanho do sistema e valida se é inteiro positivo
            n = int(input("Digite o número de equações do sistema: "))
            if n > 0:
                break
            else:
                print("Por favor, digite um número inteiro positivo.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número inteiro.")
            
    A = np.zeros((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    
    print("\nAgora, digite os coeficientes de cada equação.")
    # Loop duplo para preencher a Matriz A (Coeficientes)
    for i in range(n):
        print(f"\n--- Para a Equação {i+1} ---")
        for j in range(n):
            while True:
                try:
                    A[i, j] = float(input(f"Digite o coeficiente de x{j+1}: "))
                    break
                except ValueError:
                    print("Entrada inválida. Digite um número.")
        # Loop para preencher o vetor b (Termos independentes)
        while True:
            try:
                b[i] = float(
                    input(f"Digite o termo independente da Equação {i+1}: "))
                break
            except ValueError:
                print("Entrada inválida. Digite um número.")
    return A, b


def escolher_metodo():
    print("\nEscolha o método de resolução:")
    print("1. Eliminação de Gauss Simples")
    print("2. Eliminação de Gauss com Pivotação Parcial")
    print("3. Decomposição LU (Simples)")
    print("4. Decomposição LUP (com Pivotação)")

    while True:
        try:
            escolha = int(input("Digite sua escolha (1, 2, 3 ou 4): "))
            if escolha in [1, 2, 3, 4]:
                return escolha
            else:
                print("Escolha inválida.")
        except ValueError:
            print("Entrada inválida.")


def menu_pos_calculo():
    # ... (código inalterado)
    print("\nO que você deseja fazer agora?")
    print("1. Inserir um novo sistema de equações")
    print("2. Calcular o mesmo sistema com outro método")
    print("3. Sair do programa")
    while True:
        try:
            escolha = int(input("Digite sua escolha (1, 2 ou 3): "))
            if escolha in [1, 2, 3]:
                return escolha
            else:
                print("Escolha inválida. Por favor, digite 1, 2 ou 3.")
        except ValueError:
            print("Entrada inválida. Por favor, digite 1, 2 ou 3.")


# --- FLUXO PRINCIPAL DO PROGRAMA (atualizado) ---
A_usuario, b_usuario = None, None # Variáveis de estado para guardar os dados

while True:
    # Só pede dados novos se A_usuario for None (primeira vez ou escolha de reset)
    if A_usuario is None:
        A_usuario, b_usuario = obter_dados_do_usuario()
        
    escolha_metodo_usuario = escolher_metodo()
    solucao = None
    
    # Lógica para métodos que usam Matriz Aumentada (Gauss)
    if escolha_metodo_usuario == 1 or escolha_metodo_usuario == 2:
        # Cria a matriz aumentada juntando A e b lado a lado
        matriz_aumentada = np.hstack([A_usuario, b_usuario[:, np.newaxis]])
        print("\nO sistema inserido foi (Matriz Aumentada [A|b]):")
        print(matriz_aumentada)
        
        if escolha_metodo_usuario == 1:
            solucao = eliminacao_gauss_simples(matriz_aumentada.copy()) # Usa .copy() para não estragar a original
        else:
            solucao = gauss_com_pivoteamento(matriz_aumentada.copy())
            
    # Lógica para métodos de Fatoração (LU/LUP)
    elif escolha_metodo_usuario == 3:
        solucao = decomposicao_lu(A_usuario.copy(), b_usuario.copy())
    else:  # Escolha é 4 (Decomposição LUP)
        solucao = decomposicao_lup(A_usuario.copy(), b_usuario.copy())

    # Exibição do resultado final
    if solucao is not None:
        print("\n------------------")
        print("SOLUÇÃO ENCONTRADA")
        print("------------------")
        for i, val in enumerate(solucao):
            print(f"x{i+1} = {val:.4f}")

    # Gerenciamento do loop (repetir ou sair)
    escolha_final = menu_pos_calculo()
    if escolha_final == 1:
        # Limpa os dados para forçar o usuário a digitar tudo de novo
        A_usuario, b_usuario = None, None
        print("\n" + "="*50)
        continue
    elif escolha_final == 2:
        # Mantém A_usuario e b_usuario, apenas reinicia o loop para escolher outro método
        print("\n" + "="*50)
        print("--- Resolvendo o mesmo sistema com outro método ---")
        continue
    else:
        print("\nEncerrando o programa. Até a próxima!")
        break
