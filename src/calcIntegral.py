import numpy as np
import sympy as sp
import sys

def cabecalho(texto):
    print("\n" + "="*65)
    print(f"{texto:^65}")
    print("="*65)

def calcular_integral_analitica(funcao_str, a, b):
    x = sp.symbols('x')
    try:
        expr = sp.sympify(funcao_str)
        resultado_exato = sp.integrate(expr, (x, a, b))
        return float(resultado_exato)
    except:
        return None

def regra_trapezio(y, h):
    soma = y[0] + 2 * np.sum(y[1:-1]) + y[-1]
    return (h / 2) * soma

def simpson_13(y, h):
    n = len(y) - 1
    if n % 2 != 0:
        return None, "Erro: Requer 'n' PAR."
    soma = y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1]
    return (h / 3) * soma, None

def simpson_38(y, h):
    n = len(y) - 1
    if n % 3 != 0:
        return None, "Erro: Requer 'n' MÚLTIPLO DE 3."
    soma = y[0] + y[-1]
    for i in range(1, n):
        if i % 3 == 0:
            soma += 2 * y[i]
        else:
            soma += 3 * y[i]
    return (3 * h / 8) * soma, None

def calcular_erro(valor_exato, valor_numerico):
    if valor_exato is None or valor_exato == 0:
        return None
    return abs((valor_exato - valor_numerico) / valor_exato) * 100

def main():
    cabecalho("CALCULADORA DE INTEGRAIS NUMÉRICAS")
    print("1. Entrada por FUNÇÃO (ex: x**2, sin(x))")
    print("2. Entrada por TABELA (pontos x, y)")
    
    try:
        opcao_entrada = int(input("Escolha a opção (1 ou 2): "))
    except ValueError:
        print("Opção inválida.")
        return

    x_vals = None
    y_vals = None
    h = 0
    valor_analitico = None
    n = 0

    # --- LÓGICA DE ENTRADA ---
    if opcao_entrada == 1:
        # Modo Função
        funcao_str = input("\nDigite a função (sintaxe Python): ")
        try:
            a = float(input("Limite inferior (a): "))
            b = float(input("Limite superior (b): "))
            
            print("\nComo deseja definir a discretização?")
            print("1. Pelo número de intervalos (n)")
            print("2. Pelo tamanho do passo (h)")
            tipo_discretizacao = input("Opção (1 ou 2): ")

            if tipo_discretizacao == '1':
                n = int(input("Digite o número de intervalos (n): "))
                h = (b - a) / n
            
            elif tipo_discretizacao == '2':
                h_input = float(input("Digite o tamanho do passo (h): "))
                distancia = b - a
                n_calculado = distancia / h_input
                
                # Verifica se n é inteiro (com tolerância para float)
                if abs(n_calculado - round(n_calculado)) < 1e-9:
                    n = int(round(n_calculado))
                    h = h_input
                else:
                    n = int(round(n_calculado))
                    h = distancia / n
                    print(f"\n[AVISO] O passo h={h_input} não divide o intervalo exato.")
                    print(f"Arredondando n para {n} e ajustando h para {h:.6f}")
            else:
                print("Opção inválida.")
                return

            # Gerar dados
            x_vals = np.linspace(a, b, n + 1)
            y_vals = np.zeros(n + 1)
            
            # Converter string para função numérica
            x_sym = sp.symbols('x')
            expr = sp.sympify(funcao_str)
            f_num = sp.lambdify(x_sym, expr, 'numpy')
            y_vals = f_num(x_vals)
            
            valor_analitico = calcular_integral_analitica(funcao_str, a, b)
            
        except Exception as e:
            print(f"Erro na entrada de dados: {e}")
            return

    elif opcao_entrada == 2:
        # Modo Tabela
        print("\nDigite os valores separados por espaço.")
        try:
            str_x = input("Valores de X: ").replace(',', ' ').split()
            str_y = input("Valores de Y: ").replace(',', ' ').split()
            
            x_vals = np.array([float(v) for v in str_x])
            y_vals = np.array([float(v) for v in str_y])
            
            if len(x_vals) != len(y_vals):
                print("ERRO: Vetores X e Y de tamanhos diferentes.")
                return
            
            n = len(x_vals) - 1
            h = x_vals[1] - x_vals[0]
            
            if not np.allclose(np.diff(x_vals), h):
                print("AVISO: Intervalos de X não uniformes.")
                
        except Exception as e:
            print(f"Erro ao processar tabela: {e}")
            return

    # --- MENU DE MÉTODOS ---
    cabecalho("ESCOLHA O MÉTODO")
    print(f"Parâmetros: n = {n} | h = {h:.6f}")
    if valor_analitico is not None:
        print(f"Valor Analítico: {valor_analitico:.6f}")
    else:
        print("Valor Analítico: N/A (Entrada via Tabela)")

    print("\n1. Regra do Trapézio")
    print("2. Simpson 1/3 (Requer n PAR)")
    print("3. Simpson 3/8 (Requer n MULT. 3)")
    print("4. Calcular Todos")
    
    try:
        metodo = int(input("Opção: "))
    except:
        return

    resultados = []

    # Cálculos
    if metodo == 1 or metodo == 4:
        res = regra_trapezio(y_vals, h)
        erro = calcular_erro(valor_analitico, res)
        resultados.append(("Trapézio", res, erro))

    if metodo == 2 or metodo == 4:
        res, msg = simpson_13(y_vals, h)
        if msg:
            resultados.append(("Simpson 1/3", "ERRO: n ímpar", msg))
        else:
            erro = calcular_erro(valor_analitico, res)
            resultados.append(("Simpson 1/3", res, erro))

    if metodo == 3 or metodo == 4:
        res, msg = simpson_38(y_vals, h)
        if msg:
            resultados.append(("Simpson 3/8", "ERRO: n !% 3", msg))
        else:
            erro = calcular_erro(valor_analitico, res)
            resultados.append(("Simpson 3/8", res, erro))

    # --- EXIBIÇÃO FINAL ---
    cabecalho("RESULTADOS FINAIS")
    # Cabeçalho da tabela
    print(f"{'MÉTODO':<15} | {'CALCULADO':<15} | {'ANALÍTICO':<12} | {'ERRO %':<10}")
    print("-" * 65)

    # Formata o valor analítico para string
    str_analitico = f"{valor_analitico:.6f}" if valor_analitico is not None else "N/A"

    for nome, valor, erro_val in resultados:
        if isinstance(valor, str): # Se for mensagem de erro
            print(f"{nome:<15} | {valor:<15} | {str_analitico:<12} | -")
        else:
            str_erro = f"{erro_val:.4f}%" if erro_val is not None else "N/A"
            # Imprime as 4 colunas
            print(f"{nome:<15} | {valor:.6f}        | {str_analitico:<12} | {str_erro}")

if __name__ == "__main__":
    main()
