import numpy as np
import sympy as sp
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable, Dict, Any
import sys


# 1. CAMADA DE DOMÍNIO (Estratégias Matemáticas)

class EstrategiaIntegracao(ABC):
    """
    Interface base para algoritmos de integração numérica.
    """
    @property
    @abstractmethod
    def nome(self) -> str:
        pass

    @abstractmethod
    def calcular(self, valores_y: np.ndarray, passo_h: float) -> float:
        pass

class MetodoTrapezio(EstrategiaIntegracao):
    @property
    def nome(self) -> str:
        return "Regra do Trapézio"

    def calcular(self, valores_y: np.ndarray, passo_h: float) -> float:
        soma = valores_y[0] + 2 * np.sum(valores_y[1:-1]) + valores_y[-1]
        return (passo_h / 2) * soma

class MetodoSimpson13(EstrategiaIntegracao):
    @property
    def nome(self) -> str:
        return "Simpson 1/3"

    def calcular(self, valores_y: np.ndarray, passo_h: float) -> float:
        num_segmentos = len(valores_y) - 1
        if num_segmentos % 2 != 0:
            raise ValueError("Requer n PAR") # Mensagem curta para caber na tabela
        
        soma = (valores_y[0] 
                + 4 * np.sum(valores_y[1:-1:2]) 
                + 2 * np.sum(valores_y[2:-1:2]) 
                + valores_y[-1])
        return (passo_h / 3) * soma

class MetodoSimpson38(EstrategiaIntegracao):
    @property
    def nome(self) -> str:
        return "Simpson 3/8"

    def calcular(self, valores_y: np.ndarray, passo_h: float) -> float:
        num_segmentos = len(valores_y) - 1
        if num_segmentos % 3 != 0:
            raise ValueError("Requer n MÚLTIPLO DE 3")
        
        soma = valores_y[0] + valores_y[-1]
        for i in range(1, num_segmentos):
            fator = 2 if i % 3 == 0 else 3
            soma += fator * valores_y[i]
            
        return (3 * passo_h / 8) * soma


# 2. CAMADA DE SERVIÇO (Lógica Auxiliar)

class ServicoMatematico:
    @staticmethod
    def integral_analitica(funcao_str: str, limite_a: float, limite_b: float) -> Optional[float]:
        x = sp.symbols('x')
        try:
            expressao = sp.sympify(funcao_str.replace('^', '**'))
            resultado = sp.integrate(expressao, (x, limite_a, limite_b))
            return float(resultado)
        except Exception:
            return None

    @staticmethod
    def gerar_pontos_funcao(funcao_str: str, a: float, b: float, n: int) -> Tuple[np.ndarray, np.ndarray]:
        x_vals = np.linspace(a, b, n + 1)
        x_sym = sp.symbols('x')
        expressao = sp.sympify(funcao_str.replace('^', '**'))
        funcao_lambda = sp.lambdify(x_sym, expressao, 'numpy')
        y_vals = funcao_lambda(x_vals)
        
        if np.ndim(y_vals) == 0:
            y_vals = np.full_like(x_vals, y_vals)
            
        return x_vals, y_vals

    @staticmethod
    def calcular_erro_percentual(valor_exato: Optional[float], valor_numerico: float) -> Optional[float]:
        if valor_exato is None or abs(valor_exato) < 1e-15:
            return None
        return abs((valor_exato - valor_numerico) / valor_exato) * 100


# 3. CAMADA DE APRESENTAÇÃO

class InterfaceUsuario:
    """Gerencia I/O com foco em formatação precisa."""
    
    def exibir_cabecalho(self, texto: str):
        print("\n" + "="*80)
        print(f"{texto:^80}")
        print("="*80)

    def ler_float(self, msg: str) -> float:
        while True:
            try:
                return float(input(msg))
            except ValueError:
                print(" > Erro: Digite um número válido.")

    def ler_inteiro(self, msg: str) -> int:
        while True:
            try:
                return int(input(msg))
            except ValueError:
                print(" > Erro: Digite um número inteiro.")

    def exibir_resultados(self, resultados: List[Dict[str, Any]], valor_analitico: Optional[float]):
        """
        Exibe uma tabela estritamente formatada para evitar desalinhamento.
        """
        self.exibir_cabecalho("RESULTADOS FINAIS")
        
        # Larguras fixas das colunas
        w_metodo = 22
        w_calc = 18
        w_ana = 18
        w_erro = 15

        # Cabeçalho da tabela
        header = (f"| {'MÉTODO'.center(w_metodo)} | {'CALCULADO'.center(w_calc)} | "
                  f"{'ANALÍTICO'.center(w_ana)} | {'ERRO %'.center(w_erro)} |")
        
        divisor = "-" * len(header)
        
        print(divisor)
        print(header)
        print(divisor)

        str_analitico = f"{valor_analitico:.6f}" if valor_analitico is not None else "N/A"

        for linha in resultados:
            nome = linha['metodo']
            
            if linha['sucesso']:
                val = linha['valor']
                erro = linha['erro']
                
                # Formatação condicional para números grandes
                str_val = f"{val:.6f}" if abs(val) < 1e6 else f"{val:.4e}"
                str_erro = f"{erro:.4f}%" if erro is not None else "N/A"
                
                # Alinhamento à direita (>) para números, à esquerda (<) para texto
                print(f"| {nome:<{w_metodo}} | {str_val:>{w_calc}} | {str_analitico:>{w_ana}} | {str_erro:>{w_erro}} |")
            else:
                msg_erro = linha['mensagem']
                # Centraliza a mensagem de erro ocupando as colunas de valor e erro
                msg_formatada = f"FALHA: {msg_erro}"
                largura_restante = w_calc + w_ana + w_erro + 6 # +6 pelos separadores
                print(f"| {nome:<{w_metodo}} | {msg_formatada:^{largura_restante}} |")

        print(divisor)


# 4. ORQUESTRADOR (Main com Loop)

class AplicacaoCalculadora:
    def __init__(self):
        self.ui = InterfaceUsuario()
        self.servico = ServicoMatematico()
        self.estrategias = [MetodoTrapezio(), MetodoSimpson13(), MetodoSimpson38()]

    def _obter_dados_entrada(self) -> Tuple[np.ndarray, float, Optional[float]]:
        """Gerencia o fluxo de obter dados (seja por função ou tabela)."""
        print("\n1. Entrada por FUNÇÃO")
        print("2. Entrada por TABELA")
        print("0. Sair do Programa")
        
        while True:
            opcao = input("Escolha a entrada: ").strip()
            
            if opcao == '0':
                sys.exit(0)
            elif opcao == '1':
                return self._fluxo_entrada_funcao()
            elif opcao == '2':
                return self._fluxo_entrada_tabela()
            else:
                print("Opção inválida.")

    def _fluxo_entrada_funcao(self) -> Tuple[np.ndarray, float, Optional[float]]:
        funcao_str = input("\nDigite a função (ex: x**2): ")
        a = self.ui.ler_float("Limite inferior (a): ")
        b = self.ui.ler_float("Limite superior (b): ")

        print("\nDiscretização:")
        print("1. Por número de intervalos (n)")
        print("2. Por tamanho do passo (h)")
        modo = input("Opção: ").strip()

        n = 0
        h = 0.0

        if modo == '1':
            n = self.ui.ler_inteiro("Número de intervalos (n): ")
            h = (b - a) / n
        elif modo == '2':
            h_input = self.ui.ler_float("Tamanho do passo (h): ")
            distancia = b - a
            n_calc = distancia / h_input
            n = int(round(n_calc))
            if abs(n_calc - n) > 1e-9:
                h = distancia / n
                print(f"[Aviso] h ajustado para {h:.6f}")
            else:
                h = h_input
        else:
            print("Opção inválida. Usando n=10 padrão.")
            n = 10
            h = (b - a) / n

        x_vals, y_vals = self.servico.gerar_pontos_funcao(funcao_str, a, b, n)
        valor_analitico = self.servico.integral_analitica(funcao_str, a, b)
        return y_vals, h, valor_analitico

    def _fluxo_entrada_tabela(self) -> Tuple[np.ndarray, float, Optional[float]]:
        print("\nDigite os valores separados por espaço.")
        str_x = input("Valores de X: ").replace(',', ' ').split()
        str_y = input("Valores de Y: ").replace(',', ' ').split()
        
        x_vals = np.array([float(v) for v in str_x])
        y_vals = np.array([float(v) for v in str_y])

        if len(x_vals) != len(y_vals):
            raise ValueError("Vetores X e Y têm tamanhos diferentes.")

        h = x_vals[1] - x_vals[0]
        return y_vals, h, None

    def executar(self):
        self.ui.exibir_cabecalho("CALCULADORA DE INTEGRAIS (CLEAN ARCH)")

        # Loop Principal do Programa
        while True:
            try:
                # 1. Obtenção dos Dados (Carrega apenas uma vez)
                y_vals, h, valor_analitico = self._obter_dados_entrada()

                # 2. Loop de Métodos (Reutiliza os mesmos dados)
                while True:
                    self.ui.exibir_cabecalho("SELEÇÃO DE MÉTODO")
                    print(f"Dados atuais: n = {len(y_vals)-1} | h = {h:.6f}")
                    if valor_analitico:
                        print(f"Analítico: {valor_analitico:.6f}")
                    
                    print("\n--- Métodos ---")
                    for i, estrategia in enumerate(self.estrategias):
                        print(f"{i+1}. {estrategia.nome}")
                    print(f"{len(self.estrategias)+1}. Calcular Todos")
                    print("-" * 30)
                    print("5. Inserir Novos Dados")
                    print("0. SAIR DO PROGRAMA")
                    
                    escolha = self.ui.ler_inteiro("\nOpção: ")

                    if escolha == 0:
                        print("Encerrando...")
                        sys.exit(0)
                    
                    if escolha == 5:
                        break # Sai do loop de métodos e volta para pedir dados

                    metodos_para_executar = []
                    if 1 <= escolha <= len(self.estrategias):
                        metodos_para_executar.append(self.estrategias[escolha - 1])
                    elif escolha == len(self.estrategias) + 1:
                        metodos_para_executar = self.estrategias
                    else:
                        print("Opção inválida.")
                        continue # Volta para o menu de métodos

                    # Processamento
                    relatorio = []
                    for metodo in metodos_para_executar:
                        resultado_dict = {'metodo': metodo.nome, 'sucesso': False, 'valor': 0.0, 'erro': None, 'mensagem': ''}
                        try:
                            res_numerico = metodo.calcular(y_vals, h)
                            erro_pct = self.servico.calcular_erro_percentual(valor_analitico, res_numerico)
                            
                            resultado_dict['sucesso'] = True
                            resultado_dict['valor'] = res_numerico
                            resultado_dict['erro'] = erro_pct
                        except ValueError as ve:
                            resultado_dict['mensagem'] = str(ve)
                        except Exception as e:
                            resultado_dict['mensagem'] = str(e)
                        
                        relatorio.append(resultado_dict)

                    self.ui.exibir_resultados(relatorio, valor_analitico)
                    input("\nPressione ENTER para continuar...")

            except Exception as e:
                print(f"\n[ERRO DE EXECUÇÃO]: {e}")
                print("Reiniciando o fluxo...")

if __name__ == "__main__":
    app = AplicacaoCalculadora()
    app.executar()