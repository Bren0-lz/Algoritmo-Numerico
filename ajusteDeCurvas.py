import numpy as np
import matplotlib.pyplot as plt
import warnings

# Ignora avisos de cálculos complexos irrelevantes para o usuário
warnings.filterwarnings("ignore")

def entrada_dados():
    """Solicita os dados numéricos e os nomes dos eixos."""
    print("\n--- Entrada de Dados ---")
    try:
        x_str = input("Digite as coordenadas X separadas por espaço: ")
        y_str = input("Digite as coordenadas Y separadas por espaço: ")
        
        label_x = input("Nome para o Eixo X (ex: Altura): ").strip() or "Eixo X"
        label_y = input("Nome para o Eixo Y (ex: Peso): ").strip() or "Eixo Y"

        x = np.array([float(i) for i in x_str.split()])
        y = np.array([float(i) for i in y_str.split()])
        
        if len(x) != len(y) or len(x) < 2:
            print("ERRO: Tamanhos diferentes ou poucos pontos.")
            return None, None, None, None
            
        idx_sort = np.argsort(x)
        return x[idx_sort], y[idx_sort], label_x, label_y
    except ValueError:
        print("ERRO: Use apenas números (use ponto para decimais).")
        return None, None, None, None

def calcular_metricas(y_real, y_previsto, n_params):
    """Retorna R² e Variância Residual."""
    n = len(y_real)
    rss = np.sum((y_real - y_previsto) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    r2 = 1 - (rss / ss_tot) if ss_tot != 0 else 0.0
    gl = n - n_params
    var_res = rss / gl if gl > 0 else 0.0
    return r2, var_res

def plotar_grafico_ajuste(x, y, x_fit, y_fit, titulo, r2, var_res, lx, ly):
    """Plota os pontos e a curva ajustada."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='red', s=50, label='Dados Originais', zorder=5)
    label_legenda = f"{titulo}\n($R^2$={r2:.5f} | $\sigma^2_{{res}}$={var_res:.5f})"
    plt.plot(x_fit, y_fit, color='blue', linewidth=2, label=label_legenda)
    plt.xlabel(lx)
    plt.ylabel(ly)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    print("\n>>> Feche a janela do gráfico para continuar para as predições... <<<")
    plt.show()

def visualizar_apenas_pontos(x, y, lx, ly):
    """Plota apenas os pontos."""
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='darkred', s=70, alpha=0.8, label='Dados Coletados', zorder=5)
    plt.title("Visualização dos Pontos")
    plt.xlabel(lx)
    plt.ylabel(ly)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    plt.show()

def formatar_equacao(coeficientes):
    """Formata a equação de forma legível."""
    grau = len(coeficientes) - 1
    termos = []
    for i, c in enumerate(coeficientes):
        potencia = grau - i
        termo = f"{c:+.5f}"
        if potencia == 0: termos.append(termo)
        elif potencia == 1: termos.append(f"{termo}x")
        else: termos.append(f"{termo}x^{potencia}")
    eq = "y = " + " ".join(termos)
    return eq.replace("+", "+ ").replace("-", "- ").replace("= + ", "= ")

def menu_predicoes(poly, x_dados, y_dados, lx, ly):
    """Menu para estimar valores usando a função ajustada."""
    while True:
        print(f"\n>>> PREDIÇÕES (Baseadas no ajuste atual) <<<")
        print(f"1. Encontrar {ly} (Y) dado um valor de {lx} (X)")
        print(f"2. Encontrar {lx} (X) dado um valor de {ly} (Y)")
        print("0. Voltar ao menu principal")
        op = input("Escolha uma opção: ")
        
        if op == '0': break
        elif op == '1':
            try:
                vx = float(input(f"Digite o valor de {lx} (X): "))
                vy = poly(vx)
                print(f"\n---> Resultado: Para {lx} = {vx}, estima-se {ly} = {vy:.5f}")
                
                plt.figure(figsize=(6, 4))
                plt.title(f"Predição: Y dado X={vx}")
                plt.scatter(x_dados, y_dados, color='red', alpha=0.2, label="Dados")
                xp = np.linspace(min(min(x_dados), vx), max(max(x_dados), vx), 100)
                plt.plot(xp, poly(xp), 'b-', alpha=0.5, label="Curva Ajustada")
                plt.plot(vx, vy, 'go', markersize=10, zorder=10, label=f'Estimativa Y={vy:.2f}')
                plt.axvline(vx, color='green', linestyle=':', alpha=0.5) # Linha guia vertical
                plt.xlabel(lx); plt.ylabel(ly); plt.legend(); plt.grid(True)
                plt.show()
            except ValueError: print("Entrada inválida.")
            
        elif op == '2':
            try:
                vy_alvo = float(input(f"Digite o valor alvo de {ly} (Y): "))
                raizes = (poly - vy_alvo).roots
                raizes_reais = raizes[np.iscomplex(raizes) == False].real
                
                if len(raizes_reais) == 0:
                    print(f"\n---> Não há valor real de {lx} para este {ly} no modelo atual.")
                else:
                    print(f"\n---> Para {ly} = {vy_alvo}, possível(is) valor(es) de {lx}:")
                    for i, r in enumerate(raizes_reais):
                        print(f"   Opção {i+1}: {r:.5f}")
                    plt.figure(figsize=(6, 4))
                    plt.title(f"Predição: X dado Y={vy_alvo}")
                    plt.scatter(x_dados, y_dados, color='red', alpha=0.2, label="Dados")
                    
                    # Garante que o plot cubra as raízes encontradas, mesmo se fora dos dados originais
                    x_min_plot = min(min(x_dados), min(raizes_reais))
                    x_max_plot = max(max(x_dados), max(raizes_reais))
                    margem = (x_max_plot - x_min_plot) * 0.1 # 10% de margem
                    xp = np.linspace(x_min_plot - margem, x_max_plot + margem, 200)
                    
                    plt.plot(xp, poly(xp), 'b-', alpha=0.5, label="Curva Ajustada")
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
    x, y, lx, ly = None, None, "Eixo X", "Eixo Y"
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
            if xn is not None: x, y, lx, ly = xn, yn, lxn, lyn
        elif op == '4': visualizar_apenas_pontos(x, y, lx, ly)
        
        elif op == '3':
            try:
                grau = int(input("Grau do polinômio (1=Reta, 2=Parábola): "))
                if grau < 1: continue
                coef = np.polyfit(x, y, grau)
                poly = np.poly1d(coef)
                r2, vr = calcular_metricas(y, poly(x), grau + 1)
                print(f"\n--- MMQ Grau {grau} ---\n{formatar_equacao(coef)}")
                print(f"R²: {r2:.5f} | Var.Res: {vr:.5f}")
                plotar_grafico_ajuste(x, y, np.linspace(min(x),max(x),500), poly(np.linspace(min(x),max(x),500)), f"MMQ G{grau}", r2, vr, lx, ly)
                menu_predicoes(poly, x, y, lx, ly)
            except ValueError: print("Grau inválido.")
            
        elif op == '1': 
            plotar_grafico_ajuste(x, y, x, y, "Interpolação", 1.0, 0.0, lx, ly)
            
        elif op == '2':
            if x[0] == x[-1]: print("Erro: Reta vertical."); continue
            m = (y[-1]-y[0])/(x[-1]-x[0]); b = y[0]-m*x[0]
            poly_reta = np.poly1d([m, b])
            r2, vr = calcular_metricas(y, poly_reta(x), 2)
            print(f"\n--- Reta Extremos ---\n{formatar_equacao([m,b])}")
            plotar_grafico_ajuste(x, y, x, poly_reta(x), "Reta Extremos", r2, vr, lx, ly)
            menu_predicoes(poly_reta, x, y, lx, ly)

if __name__ == "__main__":
    menu_principal()