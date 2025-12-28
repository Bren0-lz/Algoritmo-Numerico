import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass, field


# 1. CORE (Lógica)

@dataclass
class ResultadoLinear:
    solucao: np.ndarray
    metodo: str
    sucesso: bool = True
    mensagem: str = ""
    passos: Dict[str, np.ndarray] = field(default_factory=dict)

class EstrategiaResolucao(ABC):
    @property
    @abstractmethod
    def nome(self) -> str: pass
    @abstractmethod
    def resolver(self, A: np.ndarray, b: np.ndarray) -> ResultadoLinear: pass

# Métodos de Resolução

class MetodoGauss(EstrategiaResolucao):
    @property
    def nome(self) -> str: return "Eliminação de Gauss (Pivoteamento)"
    
    def resolver(self, A: np.ndarray, b: np.ndarray) -> ResultadoLinear:
        n = len(A)
        # Trabalhar com cópia para não alterar o original
        M = np.hstack([A, b[:, np.newaxis]]).astype(float)
        
        try:
            for k in range(n - 1):
                # Pivoteamento Parcial
                pivot = np.argmax(np.abs(M[k:, k])) + k
                if np.isclose(M[pivot, k], 0): 
                    raise ValueError("Pivô nulo detectado (Sistema Singular/Indeterminado).")
                
                if pivot != k:
                    M[[k, pivot]] = M[[pivot, k]]
                
                # Eliminação
                for i in range(k + 1, n):
                    fator = M[i, k] / M[k, k]
                    M[i, k:] -= fator * M[k, k:]
            
            # Substituição Regressiva
            x = np.zeros(n)
            for i in range(n - 1, -1, -1):
                soma = np.dot(M[i, i+1:n], x[i+1:n])
                if np.isclose(M[i, i], 0):
                    raise ValueError("Divisão por zero na substituição.")
                x[i] = (M[i, n] - soma) / M[i, i]
            
            return ResultadoLinear(x, self.nome, passos={"Matriz Escalonada": M})
        except Exception as e:
            return ResultadoLinear(np.array([]), self.nome, False, str(e))

class MetodoLU(EstrategiaResolucao):
    @property
    def nome(self) -> str: return "Decomposição LU (Simples)"

    def resolver(self, A: np.ndarray, b: np.ndarray) -> ResultadoLinear:
        n = len(A)
        L = np.eye(n)
        U = A.copy().astype(float)
        try:
            for k in range(n - 1):
                for i in range(k + 1, n):
                    if np.isclose(U[k, k], 0):
                        raise ValueError("Pivô zero. Tente Gauss ou LUP.")
                    fator = U[i, k] / U[k, k]
                    L[i, k] = fator
                    U[i, k:] -= fator * U[k, k:]
            
            # Ly = b
            y = np.zeros(n)
            for i in range(n):
                y[i] = b[i] - np.dot(L[i, :i], y[:i])
            
            # Ux = y
            x = np.zeros(n)
            for i in range(n - 1, -1, -1):
                x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
                
            return ResultadoLinear(x, self.nome, passos={"Matriz L": L, "Matriz U": U})
        except Exception as e:
            return ResultadoLinear(np.array([]), self.nome, False, str(e))

class MetodoLUP(EstrategiaResolucao):
    @property
    def nome(self) -> str: return "Decomposição LUP (Robust - SciPy Fallback)"
    
    def resolver(self, A: np.ndarray, b: np.ndarray) -> ResultadoLinear:
        try:
            # Tenta usar SciPy se disponível (altíssima performance)
            from scipy.linalg import lu_factor, lu_solve
            lu, piv = lu_factor(A)
            x = lu_solve((lu, piv), b)
            return ResultadoLinear(x, self.nome, passos={"Nota": np.array([["Cálculo Otimizado via SciPy"]])})
        except ImportError:
            # Se não tiver SciPy, usa implementação manual (simplificada aqui reutilizando Gauss)
            return MetodoGauss().resolver(A, b)
        except Exception as e:
            return ResultadoLinear(np.array([]), self.nome, False, str(e))


# 2. Interface (UI)

class FormatadorVisual:
    @staticmethod
    def titulo(texto: str):
        print(f"\n{'='*60}")
        print(f"  {texto.upper()}")
        print(f"{'='*60}")

    @staticmethod
    def subtitulo(texto: str):
        print(f"\n--- {texto} ---")

    @staticmethod
    def exibir_matriz(matriz: np.ndarray, titulo: str = "Matriz"):
        print(f"\n> {titulo}:")
        try:
            linhas = matriz.shape[0]
            for i in range(linhas):
                linha_str = " | ".join([f"{val:8.4f}" for val in matriz[i]])
                print(f"  | {linha_str} |")
        except:
            print(f"  {matriz}")

    @staticmethod
    def exibir_sistema(A: np.ndarray, b: np.ndarray):
        print("\n[CONFIRMAÇÃO] O sistema interpretado foi:")
        n = len(A)
        for i in range(n):
            eq = ""
            for j in range(n):
                sinal = "+" if A[i,j] >= 0 else ""
                eq += f"{sinal} {A[i,j]:.1f}*x{j+1} "
            print(f"  Eq {i+1}: {eq} = {b[i]:.2f}")

class AssistenteEntrada:
    def ler_numero(self, msg: str, tipo=int):
        while True:
            try:
                return tipo(input(msg))
            except ValueError:
                print(" > Valor inválido. Tente novamente.")

    def ler_sistema_intuitivo(self) -> tuple:
        FormatadorVisual.subtitulo("PASSO 1: DEFINIÇÃO")
        n = self.ler_numero("Quantas variáveis (n)? ", int)
        while n <= 0:
            print(" > O número de variáveis deve ser positivo.")
            n = self.ler_numero("Quantas variáveis (n)? ", int)
            
        A = np.zeros((n, n))
        b = np.zeros(n)
        
        FormatadorVisual.subtitulo("PASSO 2: EQUAÇÕES")
        print(f"Digite os coeficientes e o resultado na mesma linha.")
        print("Ex: '2 1 10' para 2x + 1y = 10")
        
        for i in range(n):
            valido = False
            while not valido:
                entrada = input(f"\nEquação {i+1}: ").replace(',', '.').split()
                if len(entrada) != n + 1:
                    print(f" > Erro: Digite exatamente {n+1} números.")
                    continue
                try:
                    valores = [float(x) for x in entrada]
                    A[i, :] = valores[:-1]
                    b[i] = valores[-1]
                    valido = True
                except ValueError:
                    print(" > Erro: Apenas números são aceitos.")
        return A, b

    def selecionar_metodo(self, metodos: List[EstrategiaResolucao]) -> EstrategiaResolucao:
        FormatadorVisual.subtitulo("PASSO 3: ESCOLHA DO MÉTODO")
        print("Qual algoritmo você deseja usar?")
        for i, m in enumerate(metodos):
            print(f"{i+1}. {m.nome}")
        
        while True:
            try:
                op = int(input("\nOpção: "))
                if 1 <= op <= len(metodos):
                    return metodos[op-1]
                print(" > Opção inexistente.")
            except ValueError:
                print(" > Digite o número da opção.")


# 3. APLICAÇÃO (CONTROLADOR)

class AppSolverLinear:
    def __init__(self):
        self.fmt = FormatadorVisual()
        self.wizard = AssistenteEntrada()
        # Lista de estratégias disponíveis
        self.metodos = [
            MetodoGauss(), 
            MetodoLU(),
            MetodoLUP()
        ]

    def carregar_exemplo(self):
        print("\n[!] Carregando sistema exemplo...")
        A = np.array([[3, 2, -4], [2, 3, 3], [5, -3, 1]], dtype=float)
        b = np.array([3, 15, 14], dtype=float)
        return A, b

    def executar(self):
        self.fmt.titulo("Solver de Sistemas Lineares (Clean Arch)")
        
        while True:
            # MENU PRINCIPAL
            print("\n1. Novo Sistema (Passo-a-passo)")
            print("2. Carregar Exemplo (Demo)")
            print("0. Sair")
            
            escolha = input("Opção: ").strip()
            
            if escolha == '0':
                print("Encerrando...")
                break
            elif escolha == '2':
                A, b = self.carregar_exemplo()
            elif escolha == '1':
                A, b = self.wizard.ler_sistema_intuitivo()
            else:
                continue

            # Confirmação visual
            self.fmt.exibir_sistema(A, b)
            
            # LOOP DE CÁLCULO (Permite trocar método para o mesmo sistema)
            while True:
                # AQUI ESTÁ A CORREÇÃO: O usuário escolhe o método explicitamente
                metodo_escolhido = self.wizard.selecionar_metodo(self.metodos)
                
                print(f"\nCalculando via {metodo_escolhido.nome}...")
                resultado = metodo_escolhido.resolver(A, b)
                
                if resultado.sucesso:
                    self.fmt.subtitulo("RESULTADO FINAL")
                    print("Vetor Solução (x):")
                    for i, val in enumerate(resultado.solucao):
                        print(f"  x{i+1} = {val:8.4f}")
                    
                    if resultado.passos:
                        if input("\nVer matrizes intermediárias? (s/n): ").lower() == 's':
                            for k, v in resultado.passos.items():
                                self.fmt.exibir_matriz(v, k)
                else:
                    print(f"\n[ERRO MATEMÁTICO]: {resultado.mensagem}")

                # Menu pós-cálculo
                print("\n--------------------------------")
                print("1. Tentar outro método (mesmo sistema)")
                print("2. Inserir novo sistema")
                print("0. Sair")
                decisao = input("Opção: ")
                
                if decisao == '1':
                    continue # Volta para o Loop de Cálculo
                elif decisao == '2':
                    break # Sai do Loop de Cálculo, volta pro Menu Principal
                else:
                    return # Sai do programa

if __name__ == "__main__":
    AppSolverLinear().executar()