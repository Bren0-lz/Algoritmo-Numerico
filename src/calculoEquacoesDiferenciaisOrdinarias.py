import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import time
from typing import Callable, List, Tuple, Dict, Optional
from abc import ABC, abstractmethod


# 1. NÚCLEO MATEMÁTICO (Estratégia e Solver)

class MetodoNumerico(ABC):
    """Interface para estratégias de resolução numérica (Padrão Strategy)."""
    @abstractmethod
    def calcular_passo(self, func: Callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
        pass

class MetodoEuler(MetodoNumerico):
    def calcular_passo(self, func: Callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
        return y + h * func(t, y)

class MetodoEulerAperfeicoado(MetodoNumerico):
    def calcular_passo(self, func: Callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
        k1 = func(t, y)
        k2 = func(t + h, y + h * k1)
        return y + (h / 2.0) * (k1 + k2)

class MetodoRK4(MetodoNumerico):
    def calcular_passo(self, func: Callable, t: float, y: np.ndarray, h: float) -> np.ndarray:
        k1 = func(t, y)
        k2 = func(t + h/2, y + (h/2) * k1)
        k3 = func(t + h/2, y + (h/2) * k2)
        k4 = func(t + h, y + h * k3)
        return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class SolucionadorEDO:
    """Orquestrador da resolução matemática."""
    def __init__(self, metodo: MetodoNumerico):
        self.metodo = metodo

    def resolver(self, func_sistema: Callable, intervalo: Tuple[float, float], y0: List[float], h: float) -> Tuple[np.ndarray, np.ndarray]:
        t0, t_fim = intervalo
        # np.arange pode ser impreciso com floats, usamos linspace ou lógica manual protegida
        num_passos = int(np.ceil((t_fim - t0) / h))
        tempos = np.linspace(t0, t0 + num_passos * h, num_passos + 1)
        
        y0_array = np.array(y0, dtype=float)
        resultados = np.zeros((len(tempos), len(y0_array)))
        resultados[0] = y0_array

        y_atual = y0_array
        for i in range(len(tempos) - 1):
            t_atual = tempos[i]
            y_prox = self.metodo.calcular_passo(func_sistema, t_atual, y_atual, h)
            resultados[i+1] = y_prox
            y_atual = y_prox
            
        return tempos, resultados


# 2. SEGURANÇA E PARSING (SymPy)

class InterpretadorMatematico:
    """
    Substitui o perigoso 'eval' por parsing simbólico seguro.
    Permite expressões como 'sin(t)', 'pi', 'exp(y)'.
    """
    @staticmethod
    def converter_expressao_para_funcao(str_eqs: List[str], var_t: str, vars_y: List[str]) -> Callable:
        t_sym = sp.symbols(var_t)
        y_syms = sp.symbols(vars_y)
        
        try:
            exprs_sym = [sp.sympify(eq.replace('^', '**')) for eq in str_eqs]
        except sp.SympifyError as e:
            raise ValueError(f"Erro de sintaxe matemática: {e}")

        # Cria uma função Python otimizada (numpy-aware)
        funcs_lambda = [sp.lambdify((t_sym, *y_syms), expr, modules=['numpy', 'math']) for expr in exprs_sym]

        def wrapper(t_val, y_vec):
            if np.ndim(y_vec) == 0: y_vec = [y_vec]
            # Proteção contra dimensão incompatível
            if len(y_vec) != len(y_syms):
                raise ValueError("Dimensão do vetor de estado incompatível com variáveis definidas.")
            return np.array([f(t_val, *y_vec) for f in funcs_lambda])
        
        return wrapper

    @staticmethod
    def avaliar_expressao_escalar(str_expr: str) -> float:
        """Avalia inputs como 'pi/2' ou '1e-3' de forma segura."""
        try:
            expr = sp.sympify(str_expr.replace('^', '**'))
            return float(expr.evalf())
        except Exception:
            raise ValueError(f"Não foi possível interpretar o número: '{str_expr}'")

    @staticmethod
    def avaliar_analitica(str_eq: str, var_t: str, array_tempos: np.ndarray) -> np.ndarray:
        """Gera a solução exata para comparação de erro."""
        t_sym = sp.symbols(var_t)
        expr = sp.sympify(str_eq.replace('^', '**'))
        f_lambda = sp.lambdify(t_sym, expr, modules='numpy')
        
        # Se a função retornar escalar (ex: y(t)=5), cria array constante
        resultado = f_lambda(array_tempos)
        if np.ndim(resultado) == 0:
            resultado = np.full_like(array_tempos, resultado)
        return resultado


# 3. INTERFACE COM USUÁRIO (Console & Gráficos)

class InterfaceConsole:
    """
    Gerencia toda a interação com o usuário. 
    Mantém o SRP: A lógica de UI está isolada da matemática.
    """
    
    def __init__(self):
        self.mapa_metodos = {
            '1': ('Euler', MetodoEuler()),
            '2': ('Euler Aperfeiçoado', MetodoEulerAperfeicoado()),
            '3': ('Runge-Kutta 4 (RK4)', MetodoRK4())
        }

    def _ler_float(self, mensagem: str) -> float:
        while True:
            entrada = input(mensagem).strip()
            try:
                return InterpretadorMatematico.avaliar_expressao_escalar(entrada)
            except ValueError as e:
                print(f" > {e}. Tente usar ponto para decimais (ex: 0.5).")

    def _ler_texto(self, mensagem: str, padrao: str = "") -> str:
        entrada = input(mensagem).strip()
        return entrada if entrada else padrao

    def executar(self):
        print("\n" + "="*50)
        print("   SOLUCIONADOR DE EDOs")
        print("="*50)

        try:
            # 1. Configuração de Variáveis
            var_t = self._ler_texto("Nome da variável independente (ex: t): ", "t")
            
            qtd_vars = 0
            while qtd_vars < 1:
                try: 
                    qtd_vars = int(input("Número de equações/variáveis (1-3): "))
                except ValueError: pass

            vars_y = []
            print(f"Nomeie suas {qtd_vars} variáveis dependentes:")
            for i in range(qtd_vars):
                nome = self._ler_texto(f"Variável {i+1} (ex: y, v): ", f"y{i}")
                vars_y.append(nome)

            # 2. Entrada de Equações
            print(f"\nDigite as equações usando '{var_t}' e {vars_y} (Sintaxe Python/Matemática):")
            equacoes_str = []
            y0_lista = []

            for nome in vars_y:
                validado = False
                while not validado:
                    eq = input(f"d({nome})/d({var_t}) = ")
                    try:
                        # Teste rápido de sintaxe
                        InterpretadorMatematico.converter_expressao_para_funcao([eq], var_t, vars_y)
                        equacoes_str.append(eq)
                        validado = True
                    except Exception as e:
                        print(f" > Erro de sintaxe: {e}")
                
                y0_lista.append(self._ler_float(f"Valor inicial de {nome} ({nome}0): "))

            # 3. Configuração do Intervalo
            print("\n--- Configuração do Intervalo ---")
            t_inicio = self._ler_float(f"Início ({var_t}0): ")
            t_fim = self._ler_float(f"Fim ({var_t}_final): ")
            passo = self._ler_float("Passo (h): ")

            # 4. Seleção do Método
            print("\n--- Método Numérico ---")
            for k, (nome, _) in self.mapa_metodos.items():
                print(f"{k} - {nome}")
            
            opcao = '0'
            while opcao not in self.mapa_metodos:
                opcao = input("Escolha (1-3): ").strip()
            
            nome_metodo, obj_metodo = self.mapa_metodos[opcao]

            # 5. Execução
            funcao_sistema = InterpretadorMatematico.converter_expressao_para_funcao(equacoes_str, var_t, vars_y)
            solver = SolucionadorEDO(obj_metodo)
            
            print("\nCalculando...")
            tempos, resultados = solver.resolver(funcao_sistema, (t_inicio, t_fim), y0_lista, passo)

            # Exibição Resultados Numéricos
            print("\n" + "="*30)
            print(f"RESULTADO FINAL ({nome_metodo})")
            print(f"{var_t} final: {tempos[-1]:.4f}")
            for i, nome in enumerate(vars_y):
                print(f"{nome}: {resultados[-1, i]:.6f}")

            # 6. Gráficos
            self._gerar_grafico(tempos, resultados, vars_y, var_t, nome_metodo)

            # 7. Análise de Erro
            self._analisar_erro(tempos, resultados, vars_y, var_t)

        except Exception as e:
            print(f"\n[ERRO CRÍTICO]: {e}")
        finally:
            print("\nPrograma finalizado.")

    def _gerar_grafico(self, tempos, resultados, vars_y, var_t, nome_metodo):
        print("\n[Info] Gerando gráfico...")
        plt.figure(figsize=(10, 6))
        
        qtd_vars = len(vars_y)
        if qtd_vars == 1:
            plt.plot(tempos, resultados[:, 0], 'b-o', label=vars_y[0], markersize=3)
            plt.ylabel(vars_y[0])
            plt.xlabel(var_t)
            plt.title(f"Solução: {nome_metodo}")
        elif qtd_vars == 2:
            # Espaço de fase 2D
            plt.plot(resultados[:, 0], resultados[:, 1], 'r-', label='Trajetória de Fase')
            plt.xlabel(vars_y[0])
            plt.ylabel(vars_y[1])
            plt.title(f"Espaço de Fase: {nome_metodo}")
        else:
            # 3D
            ax = plt.axes(projection='3d')
            ax.plot3D(resultados[:, 0], resultados[:, 1], resultados[:, 2], 'green')
            ax.set_xlabel(vars_y[0])
            ax.set_ylabel(vars_y[1])
            ax.set_zlabel(vars_y[2])
            plt.title(f"Atrator 3D: {nome_metodo}")

        plt.grid(True)
        plt.legend()
        plt.show()

    def _analisar_erro(self, tempos, resultados, vars_y, var_t):
        if input("\nCalcular erro comparativo? (s/n): ").lower().strip() != 's':
            return

        print("\n--- Análise de Erro ---")
        print("Digite a solução exata (analítica) para comparação.")
        
        for i, nome in enumerate(vars_y):
            str_exata = input(f"{nome}_exata({var_t}) = ").strip()
            if not str_exata: continue

            try:
                y_exato = InterpretadorMatematico.avaliar_analitica(str_exata, var_t, tempos)
                valor_final_exato = y_exato[-1]
                valor_final_num = resultados[-1, i]
                
                # Evita divisão por zero
                if abs(valor_final_exato) > 1e-9:
                    erro_relativo = abs((valor_final_exato - valor_final_num) / valor_final_exato) * 100
                    texto_erro = f"{erro_relativo:.4f}%"
                else:
                    erro_abs = abs(valor_final_exato - valor_final_num)
                    texto_erro = f"{erro_abs:.6f} (Absoluto)"

                print(f" > {nome}: Numérico={valor_final_num:.5f} | Exato={valor_final_exato:.5f} | Erro={texto_erro}")
                
            except Exception as e:
                print(f"Erro ao calcular analítica para {nome}: {e}")

# ==========================================
# 4. ENTRY POINT
# ==========================================

if __name__ == "__main__":
    app = InterfaceConsole()
    app.executar()