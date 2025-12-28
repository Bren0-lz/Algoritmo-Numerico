def converter_base(numero_str, base_origem, base_destino):
    DIGITOS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # --- 1. Conversão para Decimal (Base 10) ---
    try:
        numero_decimal = int(numero_str, base_origem)
    except ValueError:
        # Erro se o número contiver dígitos inválidos para a base de origem
        raise ValueError(f"Erro: O número '{numero_str}' contém dígitos inválidos para a base {base_origem}.")
    if numero_decimal == 0:
        return "0"

    if base_destino == 10:
        return str(numero_decimal)
        
    # --- 2. Conversão de Decimal (Base 10) para a Base de Destino ---
    resultado = ""
    n = numero_decimal
    while n > 0:
        
        resto = n % base_destino
        resultado = DIGITOS[resto] + resultado
        
        n //= base_destino
        
    return resultado

def interacao_usuario():
    print("--- Conversor de Bases Numéricas (2 a 36) ---")
    
    # Solicita e formata o número
    numero_str = input("Digite o número a ser convertido (ex: 1011, F3, 45): ").strip().upper()
    
    # Solicita a base de origem
    while True:
        try:
            base_origem = int(input("Digite a base ATUAL do número (de 2 a 36): "))
            if 2 <= base_origem <= 36:
                break
            else:
                print("A base de origem deve ser um número inteiro entre 2 e 36.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número inteiro.")

    # Solicita a base de destino
    while True:
        try:
            base_destino = int(input("Digite a base FINAL para a conversão (de 2 a 36): "))
            if 2 <= base_destino <= 36:
                break
            else:
                print("A base de destino deve ser um número inteiro entre 2 e 36.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número inteiro.")

    print(f"\nTentando converter {numero_str} (Base {base_origem}) para Base {base_destino}...")
    
    # Realiza a conversão e trata possíveis erros
    try:
        resultado = converter_base(numero_str, base_origem, base_destino)
        print(f"\nResultado da Conversão:")
        print(f"{numero_str} (Base {base_origem}) = {resultado} (Base {base_destino})")
    except ValueError as e:
        print(f"\nErro na Conversão: {e}")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")

if __name__ == "__main__":
    interacao_usuario()
