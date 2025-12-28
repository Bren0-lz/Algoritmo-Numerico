import sys
from typing import List

# Constantes Globais
# Definem as regras de negócio em um único lugar
ALFABETO_NUMERICO = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BASE_MINIMA = 2
BASE_MAXIMA = 36

def converter_para_decimal(numero_str: str, base_origem: int) -> int:
    """
    Converte uma string numérica de uma base específica para inteiro decimal (Base 10).
    Lança ValueError se o número conter caracteres inválidos para a base.
    """
    try:
        return int(numero_str, base_origem)
    except ValueError:
        raise ValueError(
            f"O número '{numero_str}' contém caracteres inválidos para a base {base_origem}."
        )

def converter_decimal_para_base(numero_decimal: int, base_destino: int) -> str:
    """
    Converte um inteiro decimal (Base 10) para uma string na base de destino.
    """
    if numero_decimal == 0:
        return "0"

    digitos_resultado: List[str] = []
    
    # Processamento iterativo de divisões sucessivas
    temp_numero = numero_decimal
    while temp_numero > 0:
        resto = temp_numero % base_destino
        digitos_resultado.append(ALFABETO_NUMERICO[resto])
        temp_numero //= base_destino

    # Inverte a lista e junta para formar a string final
    return "".join(reversed(digitos_resultado))

def realizar_conversao_completa(numero_str: str, base_origem: int, base_destino: int) -> str:
    """
    Orquestra a conversão completa: Base A -> Decimal -> Base B.
    """
    # Passo 1: Normalização (Base N -> Base 10)
    valor_decimal = converter_para_decimal(numero_str, base_origem)
    
    # Passo 2: Transformação (Base 10 -> Base M)
    return converter_decimal_para_base(valor_decimal, base_destino)

# Camada de Interface do Usuário

def solicitar_inteiro_validado(mensagem: str, min_val: int, max_val: int) -> int:
    """
    Função utilitária para capturar input numérico dentro de um intervalo.
    Evita repetição de código (DRY).
    """
    while True:
        try:
            valor_input = input(mensagem)
            valor = int(valor_input)
            
            if min_val <= valor <= max_val:
                return valor
            
            print(f"Erro: O valor deve estar entre {min_val} e {max_val}.")
        except ValueError:
            print("Erro: Entrada inválida. Por favor, digite um número inteiro.")

def executar_programa():
    print("=== Conversor de Bases Numéricas Profissional ===")
    print(f"Suporte: Base {BASE_MINIMA} a {BASE_MAXIMA}\n")
    
    # Coleta de dados
    numero_entrada = input("Digite o número a ser convertido (ex: 1011, F3): ").strip().upper()
    
    base_origem = solicitar_inteiro_validado(
        f"Digite a base ATUAL do número ({BASE_MINIMA}-{BASE_MAXIMA}): ", 
        BASE_MINIMA, 
        BASE_MAXIMA
    )
    
    base_destino = solicitar_inteiro_validado(
        f"Digite a base FINAL desejada ({BASE_MINIMA}-{BASE_MAXIMA}): ", 
        BASE_MINIMA, 
        BASE_MAXIMA
    )

    print(f"\nProcessando conversão de '{numero_entrada}' (Base {base_origem})...")

    # Execução segura
    try:
        resultado = realizar_conversao_completa(numero_entrada, base_origem, base_destino)
        
        print("\n--- Resultado da Conversão ---")
        print(f"Entrada: {numero_entrada} (Base {base_origem})")
        print(f"Saída:   {resultado} (Base {base_destino})")
        
    except ValueError as erro:
        print(f"\nErro de Validação: {erro}")
    except Exception as erro:
        print(f"\nErro Inesperado: {erro}")

if __name__ == "__main__":
    try:
        executar_programa()
    except KeyboardInterrupt:
        print("\n\nOperação cancelada pelo usuário.")
        sys.exit(0)