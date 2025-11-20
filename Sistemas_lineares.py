# -*- coding: utf-8 -*-

import numpy as np

# --- MÉTODOS ANTERIORES (sem alterações) ---


def eliminacao_gauss_simples(matriz_aumentada):
    # ... (código inalterado)
    n = len(matriz_aumentada)
    for k in range(n - 1):
        if matriz_aumentada[k, k] == 0:
            print("\nERRO: Pivô zero encontrado...")
            return None
        for i in range(k + 1, n):
            multiplicador = matriz_aumentada[i, k] / matriz_aumentada[k, k]
            matriz_aumentada[i, :] = matriz_aumentada[i, :] - \
                multiplicador * matriz_aumentada[k, :]
    print("\nMatriz após a fase de eliminação (triangular superior):")
    print(np.round(matriz_aumentada, 4))
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(matriz_aumentada[i, i+1:n], x[i+1:n])
        x[i] = (matriz_aumentada[i, n] - soma) / matriz_aumentada[i, i]
    return x


def gauss_com_pivoteamento(matriz_aumentada):
    # ... (código inalterado)
    n = len(matriz_aumentada)
    for k in range(n - 1):
        max_index = k
        for i in range(k + 1, n):
            if abs(matriz_aumentada[i, k]) > abs(matriz_aumentada[max_index, k]):
                max_index = i
        if max_index != k:
            print(f"\nPivotando: Trocando linha {k+1} com linha {max_index+1}")
            matriz_aumentada[[k, max_index]] = matriz_aumentada[[max_index, k]]
        if matriz_aumentada[k, k] == 0:
            print("\nERRO: A matriz é singular...")
            return None
        for i in range(k + 1, n):
            multiplicador = matriz_aumentada[i, k] / matriz_aumentada[k, k]
            matriz_aumentada[i, :] = matriz_aumentada[i, :] - \
                multiplicador * matriz_aumentada[k, :]
    print("\nMatriz após a fase de eliminação com pivoteamento:")
    print(np.round(matriz_aumentada, 4))
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(matriz_aumentada[i, i+1:n], x[i+1:n])
        x[i] = (matriz_aumentada[i, n] - soma) / matriz_aumentada[i, i]
    return x


def decomposicao_lu(A, b):
    # ... (código inalterado)
    n = len(A)
    L = np.eye(n)
    U = A.copy()
    for k in range(n - 1):
        for i in range(k + 1, n):
            if U[k, k] == 0:
                print("\nERRO: Pivô zero encontrado. A decomposição LU simples falhou.")
                return None
            multiplicador = U[i, k] / U[k, k]
            L[i, k] = multiplicador
            U[i, :] = U[i, :] - multiplicador * U[k, :]
    print("\nMatriz L (Triangular Inferior):")
    print(np.round(L, 4))
    print("\nMatriz U (Triangular Superior):")
    print(np.round(U, 4))
    y = np.zeros(n)
    for i in range(n):
        soma = np.dot(L[i, :i], y[:i])
        y[i] = b[i] - soma
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (y[i] - soma) / U[i, i]
    return x

# --- NOVO MÉTODO: DECOMPOSIÇÃO LUP ---


def decomposicao_lup(A, b):
    """
    Resolve um sistema Ax=b usando Decomposição LUP (com Pivotação Parcial).
    """
    n = len(A)
    L = np.zeros((n, n))
    U = A.copy().astype(float)
    P = np.eye(n)  # Matriz de permutação começa como identidade

    for k in range(n - 1):
        # --- ETAPA DE PIVOTAÇÃO ---
        max_index = k + np.argmax(np.abs(U[k:, k]))
        if max_index != k:
            print(f"\nPivotando: Trocando linha {k+1} com linha {max_index+1}")
            # Troca as linhas em U, P e na parte já calculada de L
            U[[k, max_index]] = U[[max_index, k]]
            P[[k, max_index]] = P[[max_index, k]]
            L[[k, max_index]] = L[[max_index, k]]

        # --- FIM DA PIVOTAÇÃO ---

        if U[k, k] == 0:
            print("\nERRO: A matriz é singular.")
            return None

        # O resto da decomposição
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, :] -= L[i, k] * U[k, :]

    np.fill_diagonal(L, 1)  # Preenche a diagonal de L com 1s

    print("\nMatriz P (Permutação):")
    print(np.round(P, 4))
    print("\nMatriz L (Triangular Inferior):")
    print(np.round(L, 4))
    print("\nMatriz U (Triangular Superior):")
    print(np.round(U, 4))

    # --- FASE 2: RESOLUÇÃO DO SISTEMA ---
    # Passo A: Aplicar as permutações em b (Pb)
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
    print("  Resolvedor de Sistemas de Equações Lineares   ")
    print("--------------------------------------------------")
    while True:
        try:
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
    for i in range(n):
        print(f"\n--- Para a Equação {i+1} ---")
        for j in range(n):
            while True:
                try:
                    A[i, j] = float(input(f"Digite o coeficiente de x{j+1}: "))
                    break
                except ValueError:
                    print("Entrada inválida. Digite um número.")
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
A_usuario, b_usuario = None, None
while True:
    if A_usuario is None:
        A_usuario, b_usuario = obter_dados_do_usuario()
    escolha_metodo_usuario = escolher_metodo()
    solucao = None
    if escolha_metodo_usuario == 1 or escolha_metodo_usuario == 2:
        matriz_aumentada = np.hstack([A_usuario, b_usuario[:, np.newaxis]])
        print("\nO sistema inserido foi (Matriz Aumentada [A|b]):")
        print(matriz_aumentada)
        if escolha_metodo_usuario == 1:
            solucao = eliminacao_gauss_simples(matriz_aumentada.copy())
        else:
            solucao = gauss_com_pivoteamento(matriz_aumentada.copy())
    elif escolha_metodo_usuario == 3:
        solucao = decomposicao_lu(A_usuario.copy(), b_usuario.copy())
    else:  # Escolha é 4 (Decomposição LUP)
        solucao = decomposicao_lup(A_usuario.copy(), b_usuario.copy())

    if solucao is not None:
        print("\n------------------")
        print("SOLUÇÃO ENCONTRADA")
        print("------------------")
        for i, val in enumerate(solucao):
            print(f"x{i+1} = {val:.4f}")

    escolha_final = menu_pos_calculo()
    if escolha_final == 1:
        A_usuario, b_usuario = None, None
        print("\n" + "="*50)
        continue
    elif escolha_final == 2:
        print("\n" + "="*50)
        print("--- Resolvendo o mesmo sistema com outro método ---")
        continue
    else:
        print("\nEncerrando o programa. Até a próxima!")
        break
