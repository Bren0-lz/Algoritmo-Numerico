# Algoritmos Numéricos (Clean Architecture)

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-conclu%C3%ADdo-success)

Este repositório contém implementações robustas em **Python** de métodos numéricos fundamentais, desenvolvidos originalmente para a disciplina de **Algoritmos Numéricos** (Engenharia de Computação - IFF Bom Jesus) e posteriormente refatorados para padrões de engenharia de software modernos.

O objetivo vai além da matemática: demonstra a aplicação de **Clean Code**, **SOLID** e **Design Patterns** em algoritmos científicos.

## Engenharia e Arquitetura

Diferente de scripts acadêmicos comuns, este projeto foi estruturado com foco em manutenibilidade e extensibilidade:

* **Strategy Pattern:** Utilizado nos módulos de *Integração*, *Interpolação* e *Sistemas Lineares*. Permite a troca dinâmica de algoritmos (ex: mudar de Gauss para LU) sem alterar o código cliente, respeitando o princípio **Open/Closed (OCP)**.
* **Data Transfer Objects (DTOs):** Encapsulamento de resultados complexos (matrizes, vetores, logs de erro) para desacoplar o "motor matemático" da interface de usuário.
* **Segurança e Validação:** Uso de **SymPy** para parsing seguro de expressões matemáticas (evitando `eval`) e tratamento robusto de exceções (ex: pivôs nulos, matrizes singulares).
* **UX/UI no Console:** Interfaces intuitivas com "Wizards" de entrada, formatação visual de matrizes e modos de demonstração.

## Módulos do Projeto

### 1. `SistemasLineares.py` 
Solver de sistemas $Ax=b$ com arquitetura orientada a objetos.
* **Métodos:** Eliminação de Gauss (com Pivoteamento Parcial), Decomposição LU e Decomposição LUP (com fallback para SciPy).
* **Destaque:** Entrada de dados intuitiva (linha única) e visualização passo-a-passo das matrizes transformadas.

### 2. `Interpolacao.py`
Ferramenta para encontrar polinômios que se ajustam a um conjunto de dados.
* **Métodos:** Lagrange, Neville (Método Prático), Diferenças Divididas de Newton e Diferenças Finitas (Gregory-Newton).
* **Destaque:** Cálculo automático do **Erro de Truncamento** utilizando derivadas simbólicas.

### 3. `CalculoIntegrais.py`
Integração numérica para funções contínuas e dados tabulados.
* **Métodos:** Regra do Trapézio, Simpson 1/3 e Simpson 3/8.
* **Destaque:** Análise comparativa automática entre o valor numérico e a solução analítica exata.

### 4. `CalculoEquacoesDiferenciaisOrdinarias.py`
Resolução de PVI (Problemas de Valor Inicial) para EDOs.
* **Métodos:** Euler, Euler Aperfeiçoado e Runge-Kutta de 4ª Ordem (RK4).
* **Destaque:** Geração automática de gráficos comparativos com `matplotlib`.

### 5. `AjusteDeCurvas.py`
Métodos de regressão para análise de tendências em dados experimentais.
* **Métodos:** Método dos Mínimos Quadrados (MMQ) e Interpolação Linear Visual.

### 6. `ConversorDeBases.py`
Utilitário para conversão entre bases numéricas arbitrárias (Binário, Octal, Hexadecimal, etc).
* **Funcionalidade:** Conversão entre bases 2 até 36.

## Tecnologias Utilizadas

* **Python 3.12+**
* **NumPy:** Computação matricial de alta performance.
* **SciPy:** Algoritmos de álgebra linear otimizados (LAPACK).
* **SymPy:** Computação simbólica (derivadas e integrais exatas).
* **Matplotlib:** Visualização de dados e gráficos de funções.

## Como Executar

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Bren0-lz/Algoritmos-Numericos.git](https://github.com/Bren0-lz/Algoritmos-Numericos.git)
    cd Algoritmos-Numericos
    ```

2.  **Instale as dependências:**
    Recomenda-se o uso de um ambiente virtual (`venv`).
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute um módulo:**
    ```bash
    python SistemasLineares.py
    ```

## Autor

**Breno Luiz**
* [LinkedIn](https://www.linkedin.com/in/breno-luiz-silva-do-carmo-19451a243/)
* [GitHub](https://github.com/Bren0-lz)
