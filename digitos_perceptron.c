#include <stdio.h>
#include <stdlib.h>

/*10 Dígitos (0 a 9)
 *cada dígito é uma matriz 5x4 = 20 entradas
 */
#define NUM_AMOSTRAS 10
#define NUM_ENTRADAS 20
#define MAX_EPOCAS 1000

// Representação visual para facilitar o debug no console
void imprimir_digito(int entradas[NUM_ENTRADAS]) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 4; j++) {
            // Se for 1 printa #, se for -1 printa .
            printf("%c ", entradas[i * 4 + j] == 1 ? '#' : '.');
        }
        printf("\n");
    }
}

int main() {
    /* 1 = Pixel Preto, -1 = Pixel Branco
     * ordem: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 */

    int dados_treino[NUM_AMOSTRAS][NUM_ENTRADAS] = {
        // 0
        { -1,  1,  1, -1,
           1, -1, -1,  1,
           1, -1, -1,  1,
           1, -1, -1,  1,
          -1,  1,  1, -1 },
        // 1
        { -1,  1,  1, -1,
          -1, -1,  1, -1,
          -1, -1,  1, -1,
          -1, -1,  1, -1,
          -1, -1,  1, -1 },
        // 2
        { -1,  1,  1, -1,
          -1, -1, -1,  1,
          -1,  1,  1, -1,
           1, -1, -1, -1,
           1,  1,  1,  1 },
        // 3
        {  1,  1,  1, -1,
          -1, -1, -1,  1,
          -1,  1,  1, -1,
          -1, -1, -1,  1,
           1,  1,  1, -1 },
        // 4
        {  1, -1, -1,  1,
           1, -1, -1,  1,
           1,  1,  1,  1,
          -1, -1, -1,  1,
          -1, -1, -1,  1 },
        // 5
        {  1,  1,  1,  1,
           1, -1, -1, -1,
           1,  1,  1, -1,
          -1, -1, -1,  1,
           1,  1,  1, -1 },
        // 6
        { -1,  1,  1, -1,
           1, -1, -1, -1,
           1,  1,  1, -1,
           1, -1, -1,  1,
          -1,  1,  1, -1 },
        // 7
        {  1,  1,  1,  1,
          -1, -1, -1,  1,
          -1, -1,  1, -1,
          -1,  1, -1, -1,
          -1,  1, -1, -1 },
        // 8
        { -1,  1,  1, -1,
           1, -1, -1,  1,
          -1,  1,  1, -1,
           1, -1, -1,  1,
          -1,  1,  1, -1 },
        // 9
        { -1,  1,  1, -1,
           1, -1, -1,  1,
          -1,  1,  1,  1,
          -1, -1, -1,  1,
          -1,  1,  1, -1 }
    };

    /*
     * rede neural: 10 NEURÔNIOS
     * bias[10] -> 1 viés para cada neurônio (b0 a b9)
     */
    int pesos[NUM_AMOSTRAS][NUM_ENTRADAS];
    int bias[NUM_AMOSTRAS]; // b0, b1, ..., b9
    
    // inicializamos tudo com 0
    for(int n = 0; n < NUM_AMOSTRAS; n++) {
        bias[n] = 0;
        for(int w = 0; w < NUM_ENTRADAS; w++) {
            pesos[n][w] = 0;
        }
    }

    printf("=== INICIANDO TREINAMENTO ===\n");
    
    int epoca = 0;
    int houve_erro = 1;

    // Loop de Treinamento
    while(houve_erro && epoca < MAX_EPOCAS) {
        houve_erro = 0;
        
        // Para cada dígito de entrada (Amostra de treino)
        for(int i = 0; i < NUM_AMOSTRAS; i++) {
            
            // Treinamos cada um dos 10 neurônios individualmente
            for(int neuronio_id = 0; neuronio_id < NUM_AMOSTRAS; neuronio_id++) {
                
                // 1. Calcular o 'y_in' (Soma ponderada + bias) [cite: 49]
                int y_in = bias[neuronio_id];
                for(int j = 0; j < NUM_ENTRADAS; j++) {
                    y_in += dados_treino[i][j] * pesos[neuronio_id][j];
                }

                // 2. Função de Ativação (Limiar 0) [cite: 49]
                // Se y_in >= 0 saida = 1, senão saida = -1
                int y = (y_in >= 0) ? 1 : -1;

                // 3. Definir o alvo (Target)
                // Se o neurônio é o ID 5, ele deve dar 1 APENAS para a imagem do 5.
                // Para todas as outras imagens, deve dar -1.
                int t = (i == neuronio_id) ? 1 : -1;

                // 4. Atualizar pesos se houver erro [cite: 51, 52, 53]
                if (y != t) {
                    // Erro detectado! Atualiza pesos e bias.
                    // Nova Bias = Bias Antiga + Target
                    bias[neuronio_id] = bias[neuronio_id] + t;
                    
                    for(int k = 0; k < NUM_ENTRADAS; k++) {
                        // Novo Peso = Peso Antigo + (Target * Entrada)
                        pesos[neuronio_id][k] = pesos[neuronio_id][k] + (t * dados_treino[i][k]);
                    }
                    houve_erro = 1; // Marca que precisamos treinar mais
                }
            }
        }
        epoca++;
    }

    printf("Treinamento concluido em %d epocas.\n\n", epoca);

    // ==========================================
    // FASE DE TESTE / RECONHECIMENTO
    // ==========================================
    printf("=== TESTE DE RECONHECIMENTO ===\n");
    
    // Vamos testar passando cada dígito para ver se a rede reconhece
    for(int i = 0; i < NUM_AMOSTRAS; i++) {
        printf("\nEntrada: Padrao do Digito %d:\n", i);
        imprimir_digito(dados_treino[i]);
        
        printf("Classificacao da Rede:\n");
        int reconhecido = -1;

        // Passamos a imagem por todos os 10 neurônios (b0-b9)
        for(int neuronio_id = 0; neuronio_id < NUM_AMOSTRAS; neuronio_id++) {
            
            int y_in = bias[neuronio_id];
            for(int j = 0; j < NUM_ENTRADAS; j++) {
                y_in += dados_treino[i][j] * pesos[neuronio_id][j];
            }
            
            int y = (y_in >= 0) ? 1 : -1;

            if(y == 1) {
                printf("  -> Neuronio %d diz: SIM! (%d)\n", neuronio_id, y);
                reconhecido = neuronio_id;
            } else {
                // printf("  -> Neuronio %d diz: Nao. (%d)\n", neuronio_id, y);
            }
        }

        if(reconhecido == i) {
            printf("RESULTADO: SUCESSO. Reconheceu corretamente o %d.\n", i);
        } else {
            printf("RESULTADO: FALHA.\n");
        }
        printf("------------------------------------------------");
    }

    return 0;
}