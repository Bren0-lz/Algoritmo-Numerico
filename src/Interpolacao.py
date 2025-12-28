import numpy as np
import sympy as sp
import math
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass


# 1. OBJETOS DE VALOR E CONTRATOS

@dataclass
class ResultadoInterpolacao:
    valor: float
    metodo: str
    polinomio_str: str = ""
    detalhes: str = ""

class MetodoInterpolacao(ABC):
    @property
    @abstractmethod
    def nome(self) -> str:
        pass

    @abstractmethod
    def calcular(self, x_dados: np.ndarray, y_dados: np.ndarray, x_alvo: float) -> ResultadoInterpolacao:
        pass


# 2. IMPLEMENTAÇÃO DOS MÉTODOS

class MetodoLagrange(MetodoInterpolacao):
    @property
    def nome(self) -> str:
        return "Lagrange"

    def calcular(self, x_dados: np.ndarray, y_dados: np.ndarray, x_alvo: float) -> ResultadoInterpolacao:
        n = len(x_dados)
        valor_interpolado = 0.0
        
        for i in range(n):
            termo_L = 1.0
            for j in range(n):
                if i != j:
                    termo_L *= (x_alvo - x_dados[j]) / (x_dados[i] - x_dados[j])
            valor_interpolado += y_dados[i] * termo_L
            
        return ResultadoInterpolacao(
            valor=valor_interpolado, 
            metodo=self.nome,
            polinomio_str="P(x) = Σ (yi * Li(x)) [Forma de Lagrange]"
        )

class MetodoNeville(MetodoInterpolacao):
    @property
    def nome(self) -> str:
        return "Neville (Método Prático)"

    def calcular(self, x_dados: np.ndarray, y_dados: np.ndarray, x_alvo: float) -> ResultadoInterpolacao:
        n = len(y_dados)
        matriz_Q = np.zeros((n, n))
        matriz_Q[:, 0] = y_dados

        for j in range(1, n):
            for i in range(j, n):
                denominador = x_dados[i] - x_dados[i-j]
                if denominador == 0:
                    raise ValueError("Divisão por zero detectada (pontos X duplicados?).")
                
                numerador = (x_alvo - x_dados[i-j]) * matriz_Q[i, j-1] - (x_alvo - x_dados[i]) * matriz_Q[i-1, j-1]
                matriz_Q[i, j] = numerador / denominador

        return ResultadoInterpolacao(valor=matriz_Q[n-1, n-1], metodo=self.nome)

class MetodoNewtonDiferencas(MetodoInterpolacao):
    @property
    def nome(self) -> str:
        return "Newton (Diferenças Divididas)"

    def calcular(self, x_dados: np.ndarray, y_dados: np.ndarray, x_alvo: float) -> ResultadoInterpolacao:
        n = len(y_dados)
        tabela = np.zeros((n, n))
        tabela[:, 0] = y_dados

        for j in range(1, n):
            for i in range(j, n):
                denom = x_dados[i] - x_dados[i-j]
                if denom == 0: raise ValueError("Pontos X duplicados.")
                tabela[i, j] = (tabela[i, j-1] - tabela[i-1, j-1]) / denom

        coefs = tabela.diagonal()
        valor_final = coefs[0]
        termo_acumulado = 1.0
        str_poly = f"{coefs[0]:.4f}"
        
        for k in range(1, n):
            termo_acumulado *= (x_alvo - x_dados[k-1])
            valor_final += coefs[k] * termo_acumulado
            
            sinal = "+" if coefs[k] >= 0 else ""
            termos_x = "".join([f"(x - {x_dados[m]:.2f})" for m in range(k)])
            str_poly += f" {sinal} {coefs[k]:.4f}*{termos_x}"

        return ResultadoInterpolacao(valor=valor_final, metodo=self.nome, polinomio_str=str_poly)

class MetodoGregoryNewton(MetodoInterpolacao):
    @property
    def nome(self) -> str:
        return "Gregory-Newton (Diferenças Finitas)"

    def calcular(self, x_dados: np.ndarray, y_dados: np.ndarray, x_alvo: float) -> ResultadoInterpolacao:
        h = x_dados[1] - x_dados[0]
        if not np.allclose(np.diff(x_dados), h, atol=1e-9):
            raise ValueError("Requer pontos X equiespaçados.")

        n = len(y_dados)
        tabela = np.zeros((n, n))
        tabela[:, 0] = y_dados

        for j in range(1, n):
            for i in range(0, n - j):
                tabela[i, j] = tabela[i+1, j-1] - tabela[i, j-1]

        coefs = tabela[0, :]
        s_val = (x_alvo - x_dados[0]) / h
        
        valor_final = coefs[0]
        fatorial = 1.0
        termo_s = 1.0
        
        for k in range(1, n):
            termo_s *= (s_val - (k - 1))
            fatorial *= k
            valor_final += (coefs[k] * termo_s) / fatorial

        return ResultadoInterpolacao(
            valor=valor_final,
            metodo=self.nome,
            polinomio_str=f"P(s) com s={s_val:.4f}",
            detalhes=f"Passo h={h:.4f}"
        )


# 3. ANÁLISE DE ERRO

class AnalisadorErro:
    @staticmethod
    def estimar_erro(funcao_str: str, x_dados: np.ndarray, x_alvo: float) -> Tuple[Optional[float], str]:
        try:
            x_sym = sp.symbols('x')
            expr = sp.sympify(funcao_str.replace('^', '**'))
            n = len(x_dados) - 1
            
            # Derivada n+1
            derivada = sp.diff(expr, x_sym, n + 1)
            f_deriv = sp.lambdify(x_sym, sp.Abs(derivada), 'numpy')
            
            # Busca de máximo
            grid = np.linspace(np.min(x_dados), np.max(x_dados), 1000)
            vals = f_deriv(grid)
            max_val = np.max(vals) if np.ndim(vals) > 0 else float(vals)
            
            # Produtório
            prod = np.prod([abs(x_alvo - xi) for xi in x_dados])
            
            erro = (max_val / math.factorial(n + 1)) * prod
            return erro, str(derivada)
        except Exception as e:
            return None, str(e)


# 4. INTERFACE DE USUÁRIO (UI)

class InterfaceUsuario:
    def cabecalho(self, texto: str):
        print(f"\n{'='*60}\n{texto:^60}\n{'='*60}")

    def ler_pontos(self) -> Tuple[np.ndarray, np.ndarray]:
        print("\n--- Entrada de Dados ---")
        while True:
            try:
                sx = input("Valores de X: ").replace(',', ' ').split()
                sy = input("Valores de Y: ").replace(',', ' ').split()
                if not sx or not sy: continue
                
                x = np.array([float(v) for v in sx])
                y = np.array([float(v) for v in sy])
                
                if len(x) != len(y):
                    print(" > Erro: Vetores de tamanhos diferentes.")
                    continue
                if len(x) < 2:
                    print(" > Erro: Mínimo 2 pontos.")
                    continue
                return x, y
            except ValueError:
                print(" > Erro: Digite apenas números.")

    def ler_float(self, msg: str) -> float:
        while True:
            try:
                return float(input(msg))
            except ValueError:
                print(" > Inválido.")

    def menu_metodos(self, metodos: List[MetodoInterpolacao]) -> MetodoInterpolacao:
        print("\n--- Escolha o Método ---")
        for i, m in enumerate(metodos):
            print(f"{i+1}. {m.nome}")
        
        while True:
            try:
                op = int(input("Opção: "))
                if 1 <= op <= len(metodos):
                    return metodos[op-1]
                print(" > Opção inexistente.")
            except ValueError:
                print(" > Digite um número inteiro.")

    def menu_pos_calculo(self) -> str:
        print("\n" + "-"*60)
        print("O QUE DESEJA FAZER AGORA?")
        print("1. Escolher OUTRO MÉTODO (para o mesmo ponto)")
        print("2. Escolher OUTRO PONTO (para os mesmos dados)")
        print("3. Digitar NOVOS DADOS")
        print("0. Sair do programa")
        print("-" * 60)
        return input("Sua escolha: ").strip()


# 5. MAIN (com Loop Aninhado)

class AppInterpolacao:
    def __init__(self):
        self.ui = InterfaceUsuario()
        self.analisador = AnalisadorErro()
        self.estrategias = [
            MetodoLagrange(),
            MetodoNeville(),
            MetodoNewtonDiferencas(),
            MetodoGregoryNewton()
        ]

    def executar(self):
        self.ui.cabecalho("SISTEMA DE INTERPOLAÇÃO PRO")

        # LOOP 1: Ciclo de vida dos DADOS
        while True:
            try:
                x_dados, y_dados = self.ui.ler_pontos()
            except KeyboardInterrupt:
                break

            # LOOP 2: Ciclo de vida do PONTO ALVO
            while True:
                x_alvo = self.ui.ler_float(f"\nDigite o ponto para interpolar (Intervalo [{min(x_dados)}, {max(x_dados)}]): ")

                # LOOP 3: Ciclo de vida do CÁLCULO/MÉTODO
                while True:
                    metodo = self.ui.menu_metodos(self.estrategias)
                    
                    # Execução
                    try:
                        self.ui.cabecalho(f"Calculando com {metodo.nome}...")
                        res = metodo.calcular(x_dados, y_dados, x_alvo)
                        
                        print(f"\nRESULTADO: P({x_alvo}) = {res.valor:.8f}")
                        if res.polinomio_str: print(f"Polinômio: {res.polinomio_str}")
                        if res.detalhes: print(f"Nota: {res.detalhes}")

                    except ValueError as e:
                        print(f"\n[ERRO MATEMÁTICO]: {e}")
                    
                    # Análise de Erro (Opcional)
                    if input("\nCalcular erro teórico estimado? (s/n): ").lower() == 's':
                        fs = input("Função f(x) original (ex: sin(x)): ")
                        erro, msg = self.analisador.estimar_erro(fs, x_dados, x_alvo)
                        if erro is not None:
                            print(f" > Limite do Erro: {erro:.6e}")
                        else:
                            print(f" > Erro na análise: {msg}")

                    # --- MENU DE NAVEGAÇÃO ---
                    decisao = self.ui.menu_pos_calculo()

                    if decisao == '1':
                        continue # Volta para escolher MÉTODO (Loop 3)
                    elif decisao == '2':
                        break # Quebra Loop 3 -> Volta para escolher PONTO (Loop 2)
                    elif decisao == '3':
                        # Precisamos sair do Loop 2 também. Usamos um "break flag" ou return
                        # Maneira pythonica simples:
                        break 
                    elif decisao == '0':
                        print("Saindo...")
                        sys.exit(0)
                    else:
                        print("Opção inválida. Voltando ao menu de métodos.")
                
                # Se a decisão foi '3' (Novos Dados), o break acima saiu do Loop 3.
                # Agora precisamos verificar se devemos sair do Loop 2 para o 1.
                if decisao == '3':
                    break # Quebra Loop 2 -> Volta para ler DADOS (Loop 1)

if __name__ == "__main__":
    AppInterpolacao().executar()