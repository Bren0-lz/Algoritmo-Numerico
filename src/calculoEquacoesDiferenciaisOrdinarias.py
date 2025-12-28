import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# --- CÁLCULO ---
class OdeSolver:
    def __init__(self, f, t0, y0, h, t_end):
        self.f = f
        self.t0 = t0
        self.y0 = np.array(y0, dtype=float)
        self.h = h
        self.t_end = t_end
        self.times = np.arange(t0, t_end + h/100, h)
        self.times = self.times[self.times <= t_end + 1e-9]
        
        self.num_steps = len(self.times)

    def solve(self, method):
        y = np.zeros((self.num_steps, len(self.y0)))
        y[0] = self.y0
        
        step_func = getattr(self, method)
        
        for i in range(self.num_steps - 1):
            y[i+1] = step_func(self.times[i], y[i])
            
        return self.times, y

    def euler(self, t, y):
        return y + self.h * self.f(t, y)

    def euler_aperfeicoado(self, t, y):
        h = self.h
        k1 = self.f(t, y)
        k2 = self.f(t + h, y + h * k1)
        return y + (h / 2.0) * (k1 + k2)

    def rk4(self, t, y):
        h = self.h
        k1 = self.f(t, y)
        k2 = self.f(t + h/2, y + (h/2) * k1)
        k3 = self.f(t + h/2, y + (h/2) * k2)
        k4 = self.f(t + h, y + h * k3)
        return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --- INTERFACE E UTILITÁRIOS ---

def get_math_context():
    return {
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 
        'exp': np.exp, 'sqrt': np.sqrt, 'log': np.log, 
        'pi': np.pi, 'e': np.e, 'abs': np.abs
    }

def clean_input(eq_str):
    return eq_str.replace('^', '**').strip()

def input_float_expr(prompt):
    while True:
        val_str = input(prompt)
        try:
            val_str = clean_input(val_str)
            val = float(eval(val_str, {"__builtins__": None}, get_math_context()))
            return val
        except Exception:
            print(f" > Erro: Não entendi '{val_str}'. Use ponto (ex: 0.5) ou expressões (ex: 1/3).")

def user_system_wrapper(t, y_vec, equation_strings, var_t_name, var_y_names):
    context = get_math_context()
    context[var_t_name] = t
    
    if np.ndim(y_vec) == 0: y_vec = [y_vec] 
    for i, name in enumerate(var_y_names):
        if i < len(y_vec):
            context[name] = y_vec[i]
            
    derivatives = []
    for eq_str in equation_strings:
        try:
            val = eval(eq_str, {"__builtins__": None}, context)
            derivatives.append(val)
        except Exception as e:
            raise ValueError(f"Erro na equação '{eq_str}': {e}")
            
    return np.array(derivatives)

# --- PROGRAMA PRINCIPAL ---

def main():
    print("\n" + "="*50)
    print("   SOLUCIONADOR DE EDOs + ANÁLISE DE ERRO")
    print("="*50)

    # 1. DEFINIÇÃO DE NOMES
    try:
        var_t = input("Nome da variável independente (ex: t, x): ").strip() or 't'
        
        n_eq = 0
        while n_eq < 1:
            try: n_eq = int(input("Número de equações/variáveis (1-3): "))
            except: pass

        var_y_names = []
        print(f"Nomeie suas {n_eq} variáveis dependentes:")
        for i in range(n_eq):
            name = input(f"Variável {i+1} (ex: y, v, z): ").strip() or f"y{i}"
            var_y_names.append(name)

        # 2. EQUAÇÕES E CONDIÇÕES INICIAIS
        print(f"\nDigite as equações usando '{var_t}' e {var_y_names}:")
        equation_strings = []
        y0_list = []

        for name in var_y_names:
            valid = False
            while not valid:
                eq = input(f"d({name})/d({var_t}) = ")
                clean_eq = clean_input(eq)
                try:
                    ctx = {v: 1.0 for v in [var_t] + var_y_names}
                    ctx.update(get_math_context())
                    eval(clean_eq, {"__builtins__": None}, ctx)
                    equation_strings.append(clean_eq)
                    valid = True
                except:
                    print(" > Erro de sintaxe.")
            
            y0_list.append(input_float_expr(f"Valor inicial de {name} ({name}0): "))

        # 3. PARÂMETROS
        print("\n--- Configuração do Intervalo ---")
        t0 = input_float_expr(f"Início ({var_t}0): ")
        t_end = input_float_expr(f"Fim ({var_t}_final): ")
        h = input_float_expr("Passo (h): ")

        # 4. MÉTODO
        print("\n--- Método Numérico ---")
        print("1 - Euler")
        print("2 - Euler Aperfeiçoado")
        print("3 - RK4")
        
        opt = '0'
        while opt not in ['1','2','3']: opt = input("Escolha (1-3): ")
        method_map = {'1':'euler', '2':'euler_aperfeicoado', '3':'rk4'}
        
        # 5. CÁLCULO
        f_user = lambda t, y: user_system_wrapper(t, y, equation_strings, var_t, var_y_names)
        
        solver = OdeSolver(f_user, t0, y0_list, h, t_end)
        times, Y = solver.solve(method_map[opt])
        
        print("\n" + "="*30)
        print("RESULTADO NUMÉRICO FINAL")
        print(f"Tempo ({var_t}): {times[-1]:.4f}")
        for i, name in enumerate(var_y_names):
            print(f"{name}: {Y[-1, i]:.6f}")

        # --- PLOTAGEM ---
        print("\n[Info] Abrindo gráfico... O programa pausará até você fechar a janela.")
        fig = plt.figure(figsize=(10, 6))
        method_name = method_map[opt].replace('_', ' ').title()

        if n_eq == 1:
            plt.plot(times, Y[:, 0], 'b-o', label=var_y_names[0])
            plt.xlabel(f"Variável Independente ({var_t})")
            plt.ylabel(f"Variável Dependente ({var_y_names[0]})")
            plt.title(f"Solução Numérica: {method_name}")
            plt.grid(True)
            plt.legend()
        else:
            ax = fig.add_subplot(111, projection='3d')
            if n_eq == 2:
                ax.plot(Y[:, 0], Y[:, 1], times, label='Trajetória')
            else:
                ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], label='Atrator')
            ax.set_title(f"Espaço de Fase ({method_name})")
            plt.legend()

        plt.show(block=True)
        plt.close('all')
        print("\nProcessando...")
        time.sleep(1.0) 

        # --- 6. CÁLCULO DE ERRO ---
        try:
            import sys
            sys.stdin.flush()
        except: pass

        ask_erro = input("\nDeseja calcular o erro comparando com a solução exata? (s/n): ").strip().lower()
        
        if ask_erro == 's':
            print("\n" + "="*40)
            print(" ANÁLISE DE ERRO")
            
            ctx = get_math_context()
            ctx[var_t] = times 

            print(f"Digite a solução analítica (fórmula exata) para cada variável:")
            
            for i, name in enumerate(var_y_names):
                sol_str = input(f"{name}_exata({var_t}) = ")
                clean_sol = clean_input(sol_str)
                
                try:
                    y_exact = eval(clean_sol, {"__builtins__": None}, ctx)
                    if np.ndim(y_exact) == 0:
                        y_exact = np.full_like(times, y_exact)
                    
                    final_exact = y_exact[-1]
                    final_num = Y[-1, i]
                    
                    if abs(final_exact) > 1e-9:
                        erro_pct = abs((final_exact - final_num) / final_exact) * 100
                    else:
                        erro_pct = abs(final_exact - final_num)
                        print(" [Aviso: Valor exato é 0, exibindo erro absoluto]")

                    print(f"\n--- Comparação Final para '{name}' ---")
                    print(f"Numérico: {final_num:.6f}")
                    print(f"Exato:    {final_exact:.6f}")
                    print(f"Erro:     {erro_pct:.4f}%")

                except Exception as e:
                    print(f"Erro ao calcular '{name}': {e}")
        
    except Exception as e:
        print(f"\n[ERRO CRÍTICO]: {e}")
    
    finally:
        print("\n" + "-"*30)
        input("Pressione ENTER para fechar o programa...")

if __name__ == "__main__":
    main()
