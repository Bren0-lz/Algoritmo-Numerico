import numpy as np
import math
import sympy  # Biblioteca essencial para manipulação simbólica (calcular derivadas exatas)

# --- 1. MÉTODOS DE INTERPOLAÇÃO ---

def lagrange(x_points, y_points, x):
    """
    Constrói o Polinômio Interpolador de Lagrange.
    Fórmula: P(x) = Soma(yi * Li(x))
    """
    n = len(x_points)
    P = np.poly1d(0.0) # Inicializa um polinômio vazio (zero)
    
    for i in range(n):
        # Li começa como 1 (elemento neutro da multiplicação)
        Li = np.poly1d(1.0)
        
        for j in range(n):
            if i != j:
                # Constrói o termo (x - xj) / (xi - xj)
                # np.poly1d([1, -x_points[j]]) cria o binômio (1*x - xj)
                p_num = np.poly1d([1, -x_points[j]])
                p_den = x_points[i] - x_points[j]
                Li = Li * (p_num / p_den) # Multiplicação acumulativa dos termos
        
        # Soma o termo atual ponderado pelo y correspondente: P = P + yi * Li
        P = P + y_points[i] * Li
        
    # Retorna o objeto polinômio (para printar a equação) e o valor calculado em x
    return P, P(x)

def neville(x_points, y_points, x):
    """
    Método Prático (Neville/Aitken).
    Não encontra a equação do polinômio, mas calcula o valor interpolado
    usando uma tabela triangular recursiva. Ótimo para precisão numérica.
    """
    n = len(y_points)
    Q = np.zeros((n, n)) # Matriz para armazenar a tabela de diferenças
    Q[:, 0] = y_points   # A primeira coluna são os valores de Y originais
    
    for j in range(1, n): # Colunas (ordem da interpolação)
        for i in range(j, n): # Linhas
            # Fórmula de recorrência de Neville (média ponderada pelos x)
            num = (x - x_points[i-j]) * Q[i, j-1] - (x - x_points[i]) * Q[i-1, j-1]
            den = x_points[i] - x_points[i-j]
            Q[i, j] = num / den
            
    # O resultado final (melhor aproximação) fica no canto inferior direito
    return Q[n-1, n-1]

def newton_divided_differences(x_points, y_points, x):
    """
    Forma de Newton para o Polinômio Interpolador.
    Usa a tabela de Diferenças Divididas.
    P(x) = a0 + a1(x-x0) + a2(x-x0)(x-x1) ...
    """
    n = len(y_points)
    table = np.zeros((n, n))
    table[:, 0] = y_points # Coluna 0 são os Ys
    
    # --- CONSTRUÇÃO DA TABELA DE DIFERENÇAS DIVIDIDAS ---
    for j in range(1, n):
        for i in range(j, n):
            # Definição: (Delta Y) / (Delta X)
            # Note que o intervalo do denominador cresce conforme j aumenta
            table[i, j] = (table[i, j-1] - table[i-1, j-1]) / (x_points[i] - x_points[i-j])
            
    # Os coeficientes do polinômio (a0, a1, a2...) estão na DIAGONAL principal
    coeffs = table.diagonal()
    
    # --- FORMATAÇÃO DA STRING E CÁLCULO ---
    poly_str = f"{coeffs[0]:.5f}" # Começa com a0
    term_str = "" # String para acumular os termos (x-x0)(x-x1)...
    result = coeffs[0]
    term_val = 1.0 # Valor numérico acumulado dos produtos (x-xi)
    
    for k in range(1, n):
        term_val *= (x - x_points[k-1]) # Atualiza o produto (x - xi)
        result += coeffs[k] * term_val  # Soma ao resultado final
        
        # Montagem visual da equação
        term_str += f"(x - {x_points[k-1]})"
        poly_str += f" + ({coeffs[k]:.5f} * {term_str})"
        
    return poly_str, result

def gregory_newton_forward(x_points, y_points, x):
    """
    Método de Gregory-Newton (Diferenças Finitas Ascendentes).
    ATENÇÃO: Só funciona se os pontos X forem EQUIDISTANTES.
    """
    n = len(y_points)
    h = x_points[1] - x_points[0] # Passo (step) entre pontos
    
    # Verifica se todos os espaçamentos são iguais a h (com tolerância pequena)
    if not np.allclose(np.diff(x_points), h):
        print("\n" + "="*50)
        print("ERRO: Os pontos de X não são igualmente espaçados.")
        print("O método de Gregory-Newton não pode ser aplicado.")
        print("="*50)
        return None, None, None
        
    # Tabela de Diferenças Finitas (Simples, sem dividir por X)
    table = np.zeros((n, n))
    table[:, 0] = y_points
    for j in range(1, n):
        for i in range(0, n - j):
            table[i, j] = table[i+1, j-1] - table[i, j-1]
            
    coeffs = table[0, :] # Os coeficientes são a primeira linha da tabela (Diferenças Ascendentes)
    
    # Variável auxiliar 's' (normalizada) para simplificar a fórmula
    s = (x - x_points[0]) / h
    
    result = coeffs[0]
    poly_str = f"{coeffs[0]:.5f}"
    s_term_val = 1.0
    s_term_str = ""
    fact = 1.0 # Fatorial
    
    # Aplica a fórmula combinatória de Gregory-Newton
    for k in range(1, n):
        s_term_val *= (s - (k - 1)) # Produtório s(s-1)(s-2)...
        fact *= k                   # k!
        result += (coeffs[k] * s_term_val) / fact
        
        # Formatação visual da equação usando a variável 's'
        if k == 1:
            s_term_str = "s"
        else:
            s_term_str += f"(s - {k-1})"
        poly_str += f" + (({coeffs[k]:.5f} * {s_term_str}) / {k}!)"
        
    poly_str = f"P(s) = {poly_str}\nOnde s = (x - {x_points[0]}) / {h:.4f} = {s:.4f}"
    return coeffs, poly_str, result


# --- 2. FUNÇÕES AUXILIARES DE ENTRADA ---

def print_menu():
    print("\n" + "="*40)
    print("  Calculadora de Interpolação Numérica")
    print("="*40)
    print("1. Método de Lagrange")
    print("2. Método Prático (Neville)")
    print("3. Método de Newton (Diferenças Divididas)")
    print("4. Método de Gregory-Newton (Espaçamento Igual)")
    print("0. Sair")
    print("="*40)

def get_points():
    """Lê e valida os vetores X e Y."""
    while True:
        try:
            x_input = input("Digite os valores de X (separados por espaço): ")
            x_points = np.array([float(val) for val in x_input.split()])
            y_input = input("Digite os valores de Y (separados por espaço): ")
            y_points = np.array([float(val) for val in y_input.split()])
            
            if len(x_points) != len(y_points):
                print("Erro: O número de pontos X e Y deve ser o mesmo.")
                continue
            if len(x_points) < 2:
                print("Erro: Você precisa de pelo menos 2 pontos para interpolar.")
                continue
            return x_points, y_points
        except ValueError:
            print("Erro: Entrada inválida. Digite apenas números separados por espaço.")

def get_point_to_interpolate():
    while True:
        try:
            x = float(input("Digite o valor de X para interpolar (ex: 2.5): "))
            return x
        except ValueError:
            print("Erro: Entrada inválida. Digite um número.")

# --- 3. FUNÇÕES DE CÁLCULO DE ERRO TEÓRICO ---

def calcular_produtorio_erro(vetor_x, x_interpolar):
    """Calcula a parte polinomial do erro: omega(x) = (x - x0)(x - x1)...(x - xn)"""
    omega_x = 1.0
    for xi in vetor_x:
        omega_x *= (x_interpolar - xi)
    return omega_x

def estimar_erro_truncamento_maximo(vetor_x, x_interpolar, max_derivada_n_mais_1):
    """
    Calcula o limite superior do erro de truncamento.
    Fórmula: |E(x)| <= (M / (n+1)!) * |(x-x0)...(x-xn)|
    Onde M é o máximo da derivada de ordem n+1.
    """
    num_pontos = len(vetor_x)
    n = num_pontos - 1 # O grau do polinômio é n
    
    omega_x = calcular_produtorio_erro(vetor_x, x_interpolar)
    
    try:
        fatorial_n_mais_1 = math.factorial(n + 1)
        erro_max_estimado = (abs(max_derivada_n_mais_1) / fatorial_n_mais_1) * abs(omega_x)
        return erro_max_estimado, omega_x, fatorial_n_mais_1
    except Exception as e:
        print(f"Erro ao calcular erro máximo: {e}")
        return None, None, None

def processar_erro_teorico(vetor_x, x_interpolar):
    """
    Usa SymPy para realizar Cálculo Simbólico.
    Se soubermos a função original f(x), podemos derivá-la n+1 vezes
    para encontrar o majorante M e estimar o erro exato.
    """
    print("\n" + "="*40)
    print("  Cálculo do Limite do Erro Teórico (Truncamento)")
    print("="*40)
    ask = input("Deseja calcular o ERRO TEÓRICO MÁXIMO (via SymPy)? (s/n): ")
    if ask.lower() != 's': return

    print(f"\n--- Iniciando Cálculo do Erro Teórico ---")
    func_str = input("Informe a função f(x) (ex: 'sin(x)', 'exp(3*x)', 'log(x)'): ")
    n = len(vetor_x) - 1
    
    try:
        x = sympy.symbols('x') # Define 'x' como símbolo matemático
        # Converte a string do usuário (ex: "sin(x)") em objeto matemático
        f = sympy.sympify(func_str) 
        
        print(f"Grau do polinômio (n): {n}")
        
        # 1. Derivada de ordem Superior: Calcula d^(n+1)/dx^(n+1)
        # Isso seria quase impossível de fazer manualmente para funções complexas.
        f_deriv = sympy.diff(f, x, n + 1)
        print(f"A {n+1}ª derivada encontrada é: {f_deriv}")

        # 2. Encontra o máximo (M) dessa derivada no intervalo
        min_x, max_x = min(vetor_x), max(vetor_x)
        
        # 'lambdify' transforma a fórmula simbólica lenta do SymPy 
        # em uma função numérica rápida do Numpy para podermos testar valores.
        f_deriv_lambdified = sympy.lambdify(x, sympy.Abs(f_deriv), 'numpy')
        
        # Varredura: Testa 1000 pontos no intervalo para achar o pico da derivada
        x_intervalo = np.linspace(min_x, max_x, 1000)
        max_derivada_val = np.max(f_deriv_lambdified(x_intervalo))
        
        print(f"Intervalo de X considerado: [{min_x}, {max_x}]")
        print(f"Valor máximo estimado da derivada no intervalo (M): {max_derivada_val:.4f}")

        # 3. Calcula o erro final usando a fórmula matemática
        erro_max, omega, fact = estimar_erro_truncamento_maximo(vetor_x, x_interpolar, max_derivada_val)
        
        if erro_max is not None:
            print("\n" + "-"*30)
            print(f"|E({x_interpolar})| <= {erro_max:.6e}")
            print(f"Este é o LIMITE SUPERIOR do erro no ponto {x_interpolar}.")
            print("-" * 30)


    except sympy.SympifyError:
        print(f"ERRO: Não foi possível entender a função '{func_str}'.")
        print("Tente usar a notação do SymPy (ex: 'sin(x)', 'cos(x)', 'exp(x)', 'log(x)')")
    except Exception as e:
        print(f"Ocorreu um erro no SymPy: {e}")


# --- 4. FUNÇÃO PRINCIPAL ---

def main():
    while True:
        print_menu()
        choice = input("Escolha uma opção: ")

        if choice == '0':
            print("Saindo...")
            break
        
        if choice in ['1', '2', '3', '4']:
            # 1. Entrada dos dados Tabelados
            x_p, y_p = get_points()
            x_val = get_point_to_interpolate()
            
            print("\n--- Resultado da Interpolação ---")
            
            result_n = None
            
            # 2. Execução do método escolhido
            if choice == '1':
                poly, result_n = lagrange(x_p, y_p, x_val)
                print(f"Polinômio de Lagrange P(x):\n{poly}")
            
            elif choice == '2':
                print("Nota: Usando o Método Prático (Tabela Dinâmica)")
                result_n = neville(x_p, y_p, x_val)

            elif choice == '3':
                poly_str, result_n = newton_divided_differences(x_p, y_p, x_val)
                print(f"Polinômio de Newton:\nP(x) = {poly_str}")

            elif choice == '4':
                coeffs, poly_str, result_n = gregory_newton_forward(x_p, y_p, x_val)
                if result_n is not None:
                    print(f"Polinômio de Gregory-Newton:\n{poly_str}")

            # 3. Exibição do resultado
            if result_n is not None:
                print("\n" + "-"*40)
                print(f"Valor interpolado: P({x_val}) = {result_n:.7f}")
                print("-" * 40)
                
                # 4. Análise de Erro (Opcional)
                # Chama a função que usa SymPy se o usuário tiver a função real
                processar_erro_teorico(x_p, x_val)
            
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":

    main()
