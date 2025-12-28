# Algoritmos Numéricos

Este repositório contém implementações em **Python** de diversos métodos numéricos, desenvolvidos como parte da disciplina de **Algoritmos Numéricos** no curso de **Engenharia de Computação** do **Instituto Federal Fluminense: Campus Bom Jesus do Itabapoana**.

O objetivo é aplicar na prática os conceitos teóricos de cálculo numérico para resolução de problemas matemáticos computacionais.

## Estrutura do Repositório

O projeto está organizado em módulos independentes, focados em diferentes tópicos da matéria:

### 1. `AjusteDeCurvas.py`
Implementação de métodos para ajuste de curvas e regressão.
* **Tópicos:** Método dos Mínimos Quadrados (MMQ), Regressão Linear e Polinomial.

### 2. `Conversor.py`
Ferramentas para conversão de bases numéricas e representação de dados.
* **Tópicos:** Conversão Binário/Decimal/Hexadecimal.

### 3. `Interpolacao.py`
Métodos para encontrar valores desconhecidos dentro de um intervalo de dados discretos.
* **Tópicos prováveis:** Interpolação de Lagrange, Forma de Newton e Splines.

### 4. `SistemasLineares.py`
Algoritmos para resolução de sistemas de equações lineares.
* **Tópicos prováveis:** Eliminação de Gauss, Decomposição LU, Jacobi e Gauss-Seidel.

## Tecnologias Utilizadas

* **Python 3.x**
* Bibliotecas:
  * `numpy` (para manipulação de vetores e matrizes)
  * `matplotlib` (para plotagem de gráficos)

## Como Executar

Para rodar qualquer um dos scripts, certifique-se de ter o Python instalado. Clone o repositório e execute o arquivo desejado via terminal:

```bash
# Clonar o repositório
git clone [https://github.com/Bren0-lz/nome-do-repositorio.git](https://github.com/Bren0-lz/nome-do-repositorio.git)

# Entrar na pasta
cd nome-do-repositorio

# Exemplo de execução (substitua pelo arquivo que deseja testar)
python SistemasLineares.py
