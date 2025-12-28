import numpy as np
import math
import sympy 

# --- 1. MÉTODOS DE INTERPOLAÇÃO ---

def lagrange(x_points, y_points, x):
    n = len(x_points)
    P = np.poly1d(0.0)
    for i in range(n):
        Li = np.poly1d(1.0)
        for j in range(n):
            if i != j:
                p_num = np.poly1d([1, -x_points[j]])
                p_den = x_points[i] - x_points[j]
                Li = Li * (p_num / p_den)
        P = P + y_points[i] * Li
    return P, P(x)

def neville(x_points, y_points, x):
    n = len(y_points)
    Q = np.zeros((n, n))
    Q[:, 0] = y_points
    for j in range(1, n):
        for i in range(j, n):
            num = (x - x_points[i-j]) * Q[i, j-1] - (x - x_points[i]) * Q[i-1, j-1]
            den = x_points[i] - x_points[i-j]
            Q[i, j] = num / den
    return Q[n-1, n-1]

def newton_divided_differences(x_points, y_points, x):
    n = len(y_points)
    table = np.zeros((n, n))
    table[:, 0] = y_points
    for j in range(1, n):
        for i in range(j, n):
            table[i, j] = (table[i, j-1] - table[i-1, j-1]) / (x_points[i] - x_points[i-j])
    coeffs = table.diagonal()
    poly_str = f"{coeffs[0]:.5f}"
    term_str = ""
    result = coeffs[0]
    term_val = 1.0
    for k in range(1, n):
        term_val *= (x - x_points[k-1])
        result += coeffs[k] * term_val
        term_str += f"(x - {x_points[k-1]})"
        poly_str += f" + ({coeffs[k]:.5f} * {term_str})"
    return poly_str, result

def gregory_newton_forward(x_points, y_points, x):
    n = len(y_points)
    h = x_points[1] - x_points[0]
    if not np.allclose(np.diff(x_points), h):
        print("\n" + "="*50)
        print("ERRO: Os pontos de X não são igualmente espaçados.")
        print("O método de Gregory-Newton não pode ser aplicado.")
        print("="*50)
        return None, None, None
    table = np.zeros((n, n))
    table[:, 0] = y_points
    for j in range(1, n):
        for i in range(0, n - j):
            table[i, j] = table[i+1, j-1] - table[i, j-1]
    coeffs = table[0, :]
    s = (x - x_points[0]) / h
    result = coeffs[0]
    poly_str = f"{coeffs[0]:.5f}"
    s_term_val = 1.0
    s_term_str = ""
    fact = 1.0
    for k in range(1, n):
        s_term_val *= (s - (k - 1))
        fact *= k
        result += (coeffs[k] * s_term_val) / fact
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
    print("2. Método Prático")
    print("3. Método de Newton")
    print("4. Método de Gregory-Newton")
    print("0. Sair")
    print("="*40)

def get_points():
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
    """Calcula omega(x) = (x - x0)(x - x1)...(x - xn)"""
    omega_x = 1.0
    for xi in vetor_x:
        omega_x *= (x_interpolar - xi)
    return omega_x

def estimar_erro_truncamento_maximo(vetor_x, x_interpolar, max_derivada_n_mais_1):
    """Calcula o limite do erro usando a fórmula teórica."""
    num_pontos = len(vetor_x)
    n = num_pontos - 1
    
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
    Função principal para o Erro Teórico. Pede a função, usa SymPy
    para derivar e encontrar o máximo M, e depois calcula o erro.
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
        x = sympy.symbols('x')
        # Converte a string em uma expressão SymPy
        f = sympy.sympify(func_str) 
        
        print(f"Grau do polinômio (n): {n}")
        
        # 1. Calcula a (n+1)-ésima derivada simbolicamente
        f_deriv = sympy.diff(f, x, n + 1)

        # 2. Encontra o máximo (M) dessa derivada no intervalo
        min_x, max_x = min(vetor_x), max(vetor_x)
        
        # Cria uma função numérica (rápida) a partir da simbólica
        f_deriv_lambdified = sympy.lambdify(x, sympy.Abs(f_deriv), 'numpy')
        
        # Testa 1000 pontos no intervalo
        x_intervalo = np.linspace(min_x, max_x, 1000)
        max_derivada_val = np.max(f_deriv_lambdified(x_intervalo))
        
        print(f"Intervalo de X considerado: [{min_x}, {max_x}]")

        # 3. Calcula o erro máximo usando a sua função
        erro_max, omega, fact = estimar_erro_truncamento_maximo(vetor_x, x_interpolar, max_derivada_val)
        
        if erro_max is not None:
            print("\n" + "-"*30)
            print(f"|E({x_interpolar})| <= {erro_max:.6e}")
            print(f"Este é o LIMITE MÁXIMO do erro no ponto {x_interpolar}.")
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
            # 1. Obter dados
            x_p, y_p = get_points()
            x_val = get_point_to_interpolate()
            
            print("\n--- Resultado da Interpolação ---")
            
            result_n = None
            
            # 2. Calcular a interpolação P(x)
            if choice == '1':
                poly, result_n = lagrange(x_p, y_p, x_val)
                print(f"Polinômio de Lagrange P(x):\n{poly}")
            
            elif choice == '2':
                print("Nota: Usando o Método Prático")
                result_n = neville(x_p, y_p, x_val)

            elif choice == '3':
                poly_str, result_n = newton_divided_differences(x_p, y_p, x_val)
                print(f"Polinômio de Newton:\nP(x) = {poly_str}")

            elif choice == '4':
                coeffs, poly_str, result_n = gregory_newton_forward(x_p, y_p, x_val)
                if result_n is not None:
                    print(f"Polinômio de Gregory-Newton:\n{poly_str}")

            # 3. Imprimir resultado e ir para o cálculo de erro
            if result_n is not None:
                print("\n" + "-"*40)
                print(f"Valor interpolado: P({x_val}) = {result_n:.7f}")
                print("-" * 40)
                
                # 4. Chamar APENAS o cálculo de erro teórico
                processar_erro_teorico(x_p, x_val)
            
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()