import numpy as np
import matplotlib.pyplot as plt
import warnings

# --- CONFIGURAÇÃO INICIAL ---
# Ignora avisos de cálculos complexos irrelevantes para o usuário
# (Ex: avisos ao tentar calcular raiz quadrada de número negativo durante plotagem)
warnings.filterwarnings("ignore")

def entrada_dados():
    """Solicita os dados numéricos e os nomes dos eixos."""
    print("\n--- Entrada de Dados ---")
    try:
        # Lê a string inteira digitada pelo usuário
        x_str = input("Digite as coordenadas X separadas por espaço: ")
        y_str = input("Digite as coordenadas Y separadas por espaço: ")
        
        # Define nomes padrão caso o usuário dê Enter sem digitar nada
        label_x = input("Nome para o Eixo X (ex: Altura): ").strip() or "Eixo X"
        label_y = input("Nome para o Eixo Y (ex: Peso): ").strip() or "Eixo Y"

        # Converte a string "1 2 3" para array numpy [1.0, 2.0, 3.0] usando list comprehension
        x = np.array([float(i) for i in x_str.split()])
        y = np.array([float(i) for i in y_str.split()])
        
        # Validações básicas de consistência dos dados
        if len(x) != len(y) or len(x) < 2:
            print("ERRO: Tamanhos diferentes ou poucos pontos.")
            return None, None, None, None
            
        # Importante: Ordena os arrays baseado no X.
        # Se não ordenar, o gráfico de linha (plt.plot) ficará "rabiscado" indo e voltando.
        idx_sort = np.argsort(x)
        return x[idx_sort], y[idx_sort], label_x, label_y
    except ValueError:
        print("ERRO: Use apenas números (use ponto para decimais).")
        return None, None, None, None

def calcular_metricas(y_real, y_previsto, n_params):
    """
    Retorna R² (Coeficiente de Determinação) e Variância Residual.
    Métricas para avaliar a qualidade do ajuste ("Bondade do Ajuste").
    """
    n = len(y_real)
    # RSS: Soma dos Quadrados dos Resíduos (Erro entre o real e o modelo)
    rss = np.sum((y_real - y_previsto) ** 2)
    # SS_tot: Soma Total dos Quadrados (Variância total dos dados em relação à média)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    
    # Cálculo do R²: Quanto mais próximo de 1, melhor o modelo explica os dados
    r2 = 1 - (rss / ss_tot) if ss_tot != 0 else 0.0
    
    # Graus de Liberdade: Número de dados - Número de parâmetros do modelo
    gl = n - n_params
    # Variância dos resíduos (Estimativa do erro aleatório sigma^2)
    var_res = rss / gl if gl > 0 else 0.0
    return r2, var_res

def plotar_grafico_ajuste(x, y, x_fit, y_fit, titulo, r2, var_res, lx, ly):
    """Plota os pontos originais e a curva matemática ajustada."""
    plt.figure(figsize=(8, 6)) # Cria uma janela de 8x6 polegadas
    
    # Plota os pontos (scatter). zorder=5 garante que fiquem "na frente" da grade
    plt.scatter(x, y, color='red', s=50, label='Dados Originais', zorder=5)
    
    # Formata a legenda com TeX (sintaxe matemática com $$) para ficar bonito
    label_legenda = f"{titulo}\n($R^2$={r2:.5f} | $\sigma^2_{{res}}$={var_res:.5f})"
    
    # Plota a linha contínua do modelo
    plt.plot(x_fit, y_fit, color='blue', linewidth=2, label=label_legenda)
    
    plt.xlabel(lx)
    plt.ylabel(ly)
    plt.legend() # Mostra a legenda configurada acima
    plt.grid(True, linestyle='--', alpha=0.6) # Grade pontilhada
    
    # Aviso para o usuário não ficar "preso" no script
    print("\n>>> Feche a janela do gráfico para continuar para as predições... <<<")
    plt.show() # Trava a execução até fechar a janela

def visualizar_apenas_pontos(x, y, lx, ly):
    """Plota apenas os pontos (scatter plot simples)."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='darkred', s=70, alpha=0.8, label='Dados Coletados', zorder=5)
    plt.title("Visualização dos Pontos")
    plt.xlabel(lx)
    plt.ylabel(ly)
    plt.legend()
    
    # Configuração de grade mais detalhada (Major e Minor ticks)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.show()

def formatar_equacao(coeficientes):
    """
    Recebe a lista de coeficientes (ex: [2, 5, 3]) e retorna string "2x^2 + 5x + 3".
    Torna a leitura matemática amigável para o usuário.
    """
    grau = len(coeficientes) - 1
    termos = []
    for i, c in enumerate(coeficientes):
        potencia = grau - i
        termo = f"{c:+.5f}" # Formata com sinal (+ ou -) e 5 casas
        
        # Lógica para adicionar x, x^2 ou nada (termo independente)
        if potencia == 0: termos.append(termo)
        elif potencia == 1: termos.append(f"{termo}x")
        else: termos.append(f"{termo}x^{potencia}")
        
    eq = "y = " + " ".join(termos)
    # Limpeza estética para não ficar "+ -" ou "= +"
    return eq.replace("+", "+ ").replace("-", "- ").replace("= + ", "= ")

def menu_predicoes(poly, x_dados, y_dados, lx, ly):
    """
    Menu interativo para realizar estimativas com o modelo já calculado.
    Recebe 'poly', que é o objeto polinômio do Numpy pronto para calcular.
    """
    while True:
        print(f"\n>>> PREDIÇÕES (Baseadas no ajuste atual) <<<")
        print(f"1. Encontrar {ly} (Y) dado um valor de {lx} (X)")
        print(f"2. Encontrar {lx} (X) dado um valor de {ly} (Y)")
        print("0. Voltar ao menu principal")
        op = input("Escolha uma opção: ")
        
        if op == '0': break
        elif op == '1':
            # --- PREDIÇÃO DIRETA (f(x) = y) ---
            try:
                vx = float(input(f"Digite o valor de {lx} (X): "))
                vy = poly(vx) # O objeto poly funciona como uma função f(x)
                print(f"\n---> Resultado: Para {lx} = {vx}, estima-se {ly} = {vy:.5f}")
                
                # Plotagem específica da predição
                plt.figure(figsize=(6, 4))
                plt.title(f"Predição: Y dado X={vx}")
                plt.scatter(x_dados, y_dados, color='red', alpha=0.2, label="Dados")
                
                # Gera pontos para desenhar a curva suave, expandindo o range se necessário
                xp = np.linspace(min(min(x_dados), vx), max(max(x_dados), vx), 100)
                plt.plot(xp, poly(xp), 'b-', alpha=0.5, label="Curva Ajustada")
                
                # Destaca o ponto previsto
                plt.plot(vx, vy, 'go', markersize=10, zorder=10, label=f'Estimativa Y={vy:.2f}')
                plt.axvline(vx, color='green', linestyle=':', alpha=0.5) # Linha guia vertical
                plt.xlabel(lx); plt.ylabel(ly); plt.legend(); plt.grid(True)
                plt.show()
            except ValueError: print("Entrada inválida.")
            
        elif op == '2':
            # --- PREDIÇÃO INVERSA (Encontrar raízes de f(x) - y_alvo = 0) ---
            try:
                vy_alvo = float(input(f"Digite o valor alvo de {ly} (Y): "))
                
                # Subtrai o alvo do polinômio e acha as raízes
                # Ex: 2x + 4 = 10  ->  2x + 4 - 10 = 0
                raizes = (poly - vy_alvo).roots
                
                # Filtra apenas números reais (ignora raízes complexas imaginárias)
                raizes_reais = raizes[np.iscomplex(raizes) == False].real
                
                if len(raizes_reais) == 0:
                    print(f"\n---> Não há valor real de {lx} para este {ly} no modelo atual.")
                else:
                    print(f"\n---> Para {ly} = {vy_alvo}, possível(is) valor(es) de {lx}:")
                    for i, r in enumerate(raizes_reais):
                        print(f"    Opção {i+1}: {r:.5f}")
                    
                    # --- Plotagem da predição inversa ---
                    plt.figure(figsize=(6, 4))
                    plt.title(f"Predição: X dado Y={vy_alvo}")
                    plt.scatter(x_dados, y_dados, color='red', alpha=0.2, label="Dados")
                    
                    # Garante que o plot cubra as raízes encontradas, mesmo se fora dos dados originais
                    x_min_plot = min(min(x_dados), min(raizes_reais))
                    x_max_plot = max(max(x_dados), max(raizes_reais))
                    margem = (x_max_plot - x_min_plot) * 0.1 # 10% de margem visual
                    xp = np.linspace(x_min_plot - margem, x_max_plot + margem, 200)
                    
                    plt.plot(xp, poly(xp), 'b-', alpha=0.5, label="Curva Ajustada")
                    # Linha horizontal no Y alvo
                    plt.axhline(vy_alvo, color='orange', linestyle='--', label=f'Alvo Y={vy_alvo}')
                    
                    # Plota todas as soluções encontradas
                    for i, r in enumerate(raizes_reais):
                        plt.plot(r, vy_alvo, 'go', markersize=8, zorder=10, label=f'Solução X={r:.2f}' if i==0 else "")
                        plt.axvline(r, color='green', linestyle=':', alpha=0.3)

                    plt.xlabel(lx); plt.ylabel(ly); plt.legend(); plt.grid(True)
                    plt.show()
                    # ------------------------------------------

            except ValueError: print("Entrada inválida.")

def menu_principal():
    """Controla o fluxo principal do programa."""
    x, y, lx, ly = None, None, "Eixo X", "Eixo Y"
    
    # Loop até conseguir dados válidos
    while x is None: x, y, lx, ly = entrada_dados()
    
    while True:
        print("\n=== FERRAMENTA DE AJUSTE DE CURVAS ===")
        print(f"Dados atuais: {len(x)} pontos | X: {lx} | Y: {ly}")
        print("1. Interpolação Linear (Apenas visual)")
        print("2. Reta (Primeiro e Último ponto)")
        print("3. MMQ (Regressão Polinomial - Melhor Ajuste)")
        print("-" * 30)
        print("4. Visualizar apenas Pontos")
        print("5. Inserir novos dados")
        print("0. Sair")
        op = input("Opção: ")
        
        if op == '0': break
        elif op == '5': 
            xn, yn, lxn, lyn = entrada_dados()
            # Só atualiza se a entrada for válida
            if xn is not None: x, y, lx, ly = xn, yn, lxn, lyn
        elif op == '4': visualizar_apenas_pontos(x, y, lx, ly)
        
        elif op == '3':
            # --- MÉTODO DOS MÍNIMOS QUADRADOS (MMQ) ---
            try:
                grau = int(input("Grau do polinômio (1=Reta, 2=Parábola): "))
                if grau < 1: continue
                
                # np.polyfit calcula os coeficientes que minimizam o erro quadrático
                coef = np.polyfit(x, y, grau)
                # np.poly1d transforma os coeficientes em uma função usável f(x)
                poly = np.poly1d(coef)
                
                # Calcula métricas estatísticas
                r2, vr = calcular_metricas(y, poly(x), grau + 1)
                
                print(f"\n--- MMQ Grau {grau} ---\n{formatar_equacao(coef)}")
                print(f"R²: {r2:.5f} | Var.Res: {vr:.5f}")
                
                # Plota usando uma linha suave (500 pontos)
                plotar_grafico_ajuste(x, y, np.linspace(min(x),max(x),500), poly(np.linspace(min(x),max(x),500)), f"MMQ G{grau}", r2, vr, lx, ly)
                
                # Vai para o menu de predições com o modelo gerado
                menu_predicoes(poly, x, y, lx, ly)
            except ValueError: print("Grau inválido.")
            
        elif op == '1': 
            # Interpolação simples: apenas liga os pontos originais
            plotar_grafico_ajuste(x, y, x, y, "Interpolação", 1.0, 0.0, lx, ly)
            
        elif op == '2':
            # --- RETA PASSANDO PELOS EXTREMOS ---
            # Útil em física básica, não usa estatística avançada
            if x[0] == x[-1]: print("Erro: Reta vertical."); continue
            
            # Cálculo manual da inclinação (m) e intercepto (b)
            m = (y[-1]-y[0])/(x[-1]-x[0]); b = y[0]-m*x[0]
            
            poly_reta = np.poly1d([m, b])
            r2, vr = calcular_metricas(y, poly_reta(x), 2)
            
            print(f"\n--- Reta Extremos ---\n{formatar_equacao([m,b])}")
            plotar_grafico_ajuste(x, y, x, poly_reta(x), "Reta Extremos", r2, vr, lx, ly)
            menu_predicoes(poly_reta, x, y, lx, ly)

if __name__ == "__main__":
    # Ponto de entrada do script
    menu_principal()
