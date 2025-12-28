import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Tuple, Optional

# Configurações globais
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

def obter_dados_usuario() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str]:
    """
    Solicita e processa os dados numéricos e rótulos dos eixos inseridos pelo utilizador.
    """
    print("\n--- Entrada de Dados ---")
    try:
        entrada_x = input("Digite as coordenadas X separadas por espaço: ")
        entrada_y = input("Digite as coordenadas Y separadas por espaço: ")
        
        rotulo_x = input("Nome para o Eixo X (ex: Altura): ").strip() or "Eixo X"
        rotulo_y = input("Nome para o Eixo Y (ex: Peso): ").strip() or "Eixo Y"

        # Converte strings para arrays numéricos
        array_x = np.array([float(i) for i in entrada_x.split()])
        array_y = np.array([float(i) for i in entrada_y.split()])
        
        if len(array_x) != len(array_y) or len(array_x) < 2:
            print("ERRO: Tamanhos diferentes ou poucos pontos (mínimo 2).")
            return None, None, "", ""
            
        # Ordenar dados com base em X para garantir que o gráfico de linha siga a ordem correta
        indices_ordenados = np.argsort(array_x)
        return array_x[indices_ordenados], array_y[indices_ordenados], rotulo_x, rotulo_y

    except ValueError:
        print("ERRO: Certifique-se de usar apenas números (use ponto para decimais).")
        return None, None, "", ""

def calcular_metricas(y_real: np.ndarray, y_previsto: np.ndarray, num_parametros: int) -> Tuple[float, float]:
    """
    Calcula o coeficiente de determinação (R²) e a variância residual.
    """
    num_amostras = len(y_real)
    soma_residuos_quad = np.sum((y_real - y_previsto) ** 2)
    soma_total_quad = np.sum((y_real - np.mean(y_real)) ** 2)
    
    r_quadrado = 1 - (soma_residuos_quad / soma_total_quad) if soma_total_quad != 0 else 0.0
    
    graus_liberdade = num_amostras - num_parametros
    variancia_residual = soma_residuos_quad / graus_liberdade if graus_liberdade > 0 else 0.0
    
    return r_quadrado, variancia_residual

def formatar_equacao_polinomio(coeficientes: np.ndarray) -> str:
    """
    Gera uma string formatada da equação polinomial.
    """
    grau = len(coeficientes) - 1
    lista_termos = []
    
    for i, coef in enumerate(coeficientes):
        potencia = grau - i
        termo_formatado = f"{coef:+.5f}"
        
        if potencia == 0:
            lista_termos.append(termo_formatado)
        elif potencia == 1:
            lista_termos.append(f"{termo_formatado}x")
        else:
            lista_termos.append(f"{termo_formatado}x^{potencia}")
            
    equacao = "y = " + " ".join(lista_termos)
    return equacao.replace("+", "+ ").replace("-", "- ").replace("= + ", "= ")

# --- Funções de Plotagem ---

def configurar_grafico(titulo: str, rotulo_x: str, rotulo_y: str):
    """Aplica configurações padrão ao gráfico atual."""
    plt.xlabel(rotulo_x)
    plt.ylabel(rotulo_y)
    plt.title(titulo)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.minorticks_on()

def plotar_ajuste(x: np.ndarray, y: np.ndarray, x_ajuste: np.ndarray, y_ajuste: np.ndarray, 
                  titulo: str, equacao: str, r2: float, var_res: float, rot_x: str, rot_y: str):
    """Exibe o gráfico com os dados originais, a curva ajustada e a equação na legenda."""
    plt.figure(figsize=(8, 6))
    
    plt.scatter(x, y, color='red', s=50, label='Dados Originais', zorder=5)
    
    # Monta a legenda com Título, Equação e Métricas
    # Usamos \\sigma para evitar o SyntaxWarning
    texto_legenda = (f"{titulo}\n"
                     f"{equacao}\n"
                     f"($R^2$={r2:.4f} | $\\sigma^2_{{res}}$={var_res:.4f})")
    
    plt.plot(x_ajuste, y_ajuste, color='blue', linewidth=2, label=texto_legenda)
    
    configurar_grafico("Ajuste de Curva", rot_x, rot_y)
    print("\n>>> Feche a janela do gráfico para continuar... <<<")
    plt.show()

def plotar_apenas_pontos(x: np.ndarray, y: np.ndarray, rot_x: str, rot_y: str):
    """Exibe apenas os pontos coletados."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='darkred', s=70, alpha=0.8, label='Dados Coletados')
    configurar_grafico("Visualização dos Pontos", rot_x, rot_y)
    plt.show()

def plotar_predicao_y(x_dados, y_dados, funcao_poli, valor_x, valor_y, rot_x, rot_y):
    """Exibe visualmente a predição de Y para um dado X."""
    plt.figure(figsize=(6, 4))
    
    plt.scatter(x_dados, y_dados, color='red', alpha=0.2, label="Dados Originais")
    
    margem = (max(x_dados) - min(x_dados)) * 0.1
    x_plot = np.linspace(min(min(x_dados), valor_x) - margem, max(max(x_dados), valor_x) + margem, 100)
    plt.plot(x_plot, funcao_poli(x_plot), 'b-', alpha=0.5, label="Modelo Ajustado")
    
    plt.plot(valor_x, valor_y, 'go', markersize=10, zorder=10, label=f'Estimativa: {valor_y:.2f}')
    plt.axvline(valor_x, color='green', linestyle=':', alpha=0.5)
    
    configurar_grafico(f"Predição: Y dado X={valor_x}", rot_x, rot_y)
    plt.show()

def plotar_predicao_x(x_dados, y_dados, funcao_poli, valor_y_alvo, raizes_reais, rot_x, rot_y):
    """Exibe visualmente a busca de X para um dado Y alvo."""
    plt.figure(figsize=(6, 4))
    plt.scatter(x_dados, y_dados, color='red', alpha=0.2, label="Dados Originais")
    
    x_min = min(min(x_dados), min(raizes_reais))
    x_max = max(max(x_dados), max(raizes_reais))
    margem = (x_max - x_min) * 0.1
    
    x_plot = np.linspace(x_min - margem, x_max + margem, 200)
    plt.plot(x_plot, funcao_poli(x_plot), 'b-', alpha=0.5, label="Modelo Ajustado")
    
    plt.axhline(valor_y_alvo, color='orange', linestyle='--', label=f'Alvo Y={valor_y_alvo}')
    
    for i, raiz in enumerate(raizes_reais):
        texto = f'Solução X={raiz:.2f}' if i == 0 else ""
        plt.plot(raiz, valor_y_alvo, 'go', markersize=8, zorder=10, label=texto)
        plt.axvline(raiz, color='green', linestyle=':', alpha=0.3)
        
    configurar_grafico(f"Predição: X dado Y={valor_y_alvo}", rot_x, rot_y)
    plt.show()

# --- Lógica de Negócio ---

def realizar_predicao_y(funcao_poli, x_dados, y_dados, rot_x, rot_y):
    try:
        entrada_valor = float(input(f"Digite o valor de {rot_x} (X): "))
        resultado_y = funcao_poli(entrada_valor)
        print(f"\n---> Resultado: Para {rot_x} = {entrada_valor}, estima-se {rot_y} = {resultado_y:.5f}")
        plotar_predicao_y(x_dados, y_dados, funcao_poli, entrada_valor, resultado_y, rot_x, rot_y)
    except ValueError:
        print("Entrada inválida.")

def realizar_predicao_x(funcao_poli, x_dados, y_dados, rot_x, rot_y):
    try:
        valor_y_alvo = float(input(f"Digite o valor alvo de {rot_y} (Y): "))
        raizes = (funcao_poli - valor_y_alvo).roots
        raizes_reais = raizes[np.iscomplex(raizes) == False].real
        
        if len(raizes_reais) == 0:
            print(f"\n---> Não há valor real de {rot_x} para este {rot_y} no modelo atual.")
        else:
            print(f"\n---> Para {rot_y} = {valor_y_alvo}, valor(es) possível(is) de {rot_x}:")
            for i, raiz in enumerate(raizes_reais):
                print(f"   Opção {i+1}: {raiz:.5f}")
            plotar_predicao_x(x_dados, y_dados, funcao_poli, valor_y_alvo, raizes_reais, rot_x, rot_y)
    except ValueError:
        print("Entrada inválida.")

def menu_predicoes(funcao_poli: np.poly1d, x: np.ndarray, y: np.ndarray, rot_x: str, rot_y: str):
    """Sub-menu para realizar inferências com o modelo ajustado."""
    while True:
        print(f"\n>>> PREDIÇÕES (Baseadas no ajuste atual) <<<")
        print(f"1. Encontrar {rot_y} (Y) dado um valor de {rot_x} (X)")
        print(f"2. Encontrar {rot_x} (X) dado um valor de {rot_y} (Y)")
        print("0. Voltar ao menu principal")
        opcao = input("Escolha uma opção: ")
        if opcao == '0': break
        elif opcao == '1': realizar_predicao_y(funcao_poli, x, y, rot_x, rot_y)
        elif opcao == '2': realizar_predicao_x(funcao_poli, x, y, rot_x, rot_y)
        else: print("Opção inválida.")

def executar_mmq(x: np.ndarray, y: np.ndarray, rot_x: str, rot_y: str):
    """Executa o Método dos Mínimos Quadrados."""
    try:
        grau = int(input("Grau do polinômio (1=Reta, 2=Parábola, etc): "))
        if grau < 1:
            print("Grau deve ser maior ou igual a 1.")
            return

        coeficientes = np.polyfit(x, y, grau)
        polinomio = np.poly1d(coeficientes)
        
        # Formata a equação aqui para usar tanto no print quanto no gráfico
        equacao_texto = formatar_equacao_polinomio(coeficientes)
        
        r2, var_residual = calcular_metricas(y, polinomio(x), grau + 1)
        
        print(f"\n--- MMQ Grau {grau} ---")
        print(equacao_texto)
        print(f"R²: {r2:.5f} | Var.Res: {var_residual:.5f}")
        
        x_plot = np.linspace(min(x), max(x), 500)
        
        # Passamos equacao_texto para o gráfico agora
        plotar_ajuste(x, y, x_plot, polinomio(x_plot), f"MMQ G{grau}", equacao_texto, r2, var_residual, rot_x, rot_y)
        
        menu_predicoes(polinomio, x, y, rot_x, rot_y)
        
    except ValueError:
        print("Grau inválido (deve ser um número inteiro).")

def executar_reta_extremos(x: np.ndarray, y: np.ndarray, rot_x: str, rot_y: str):
    """Cria uma reta ligando estritamente o primeiro e o último ponto."""
    if x[0] == x[-1]:
        print("Erro: Reta vertical impossível.")
        return

    inclinacao = (y[-1] - y[0]) / (x[-1] - x[0])
    intercepto = y[0] - inclinacao * x[0]
    
    polinomio_reta = np.poly1d([inclinacao, intercepto])
    equacao_texto = formatar_equacao_polinomio([inclinacao, intercepto])
    
    r2, var_residual = calcular_metricas(y, polinomio_reta(x), 2)
    
    print(f"\n--- Reta Extremos ---")
    print(equacao_texto)
    
    plotar_ajuste(x, y, x, polinomio_reta(x), "Reta Extremos", equacao_texto, r2, var_residual, rot_x, rot_y)
    menu_predicoes(polinomio_reta, x, y, rot_x, rot_y)

def menu_principal():
    """Controlador principal."""
    x, y = None, None
    rotulo_x, rotulo_y = "Eixo X", "Eixo Y"

    while x is None:
        x, y, rotulo_x, rotulo_y = obter_dados_usuario()
    
    while True:
        print("\n=== FERRAMENTA DE AJUSTE DE CURVAS ===")
        print(f"Dados atuais: {len(x)} pontos | X: {rotulo_x} | Y: {rotulo_y}")
        print("1. Interpolação Linear (Visual)")
        print("2. Reta (Primeiro e Último ponto)")
        print("3. MMQ (Regressão Polinomial)")
        print("-" * 30)
        print("4. Visualizar apenas Pontos")
        print("5. Inserir novos dados")
        print("0. Sair")
        
        opcao = input("Opção: ")
        
        if opcao == '0':
            print("Encerrando aplicação...")
            break
        elif opcao == '5': 
            novo_x, novo_y, novo_rot_x, novo_rot_y = obter_dados_usuario()
            if novo_x is not None: x, y, rotulo_x, rotulo_y = novo_x, novo_y, novo_rot_x, novo_rot_y
        elif opcao == '4':
            plotar_apenas_pontos(x, y, rotulo_x, rotulo_y)
        elif opcao == '3':
            executar_mmq(x, y, rotulo_x, rotulo_y)
        elif opcao == '1': 
            plotar_ajuste(x, y, x, y, "Interpolação", "Conexão Direta", 1.0, 0.0, rotulo_x, rotulo_y)
        elif opcao == '2':
            executar_reta_extremos(x, y, rotulo_x, rotulo_y)
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    menu_principal()