//OBSERVAÇÃO: Recomendo aumentar o tamanho do terminal após rodar o .out gerado, 
//se não a formatação das matrizes ficarão desalinhadas.


#include <stdio.h>
#include <string.h> // Para strcpy

// --- 1. Definições de Dados ---

struct LogicalFunction {
    char name[16];
    int outputs[4]; // Saídas para (0,0), (0,1), (1,0), (1,1)
};

struct LogicalFunction all_16_functions[16] = {
    {"F0_FALSE",    {0, 0, 0, 0}},
    {"F1_AND",      {0, 0, 0, 1}},
    {"F2_A_not_B",  {0, 0, 1, 0}},
    {"F3_A",        {0, 0, 1, 1}},
    {"F4_not_A_B",  {0, 1, 0, 0}},
    {"F5_B",        {0, 1, 0, 1}},
    {"F6_XOR",      {0, 1, 1, 0}},
    {"F7_OR",       {0, 1, 1, 1}},
    {"F8_NOR",      {1, 0, 0, 0}},
    {"F9_XNOR",     {1, 0, 0, 1}},
    {"F10_not_B",   {1, 0, 1, 0}},
    {"F11_B_imp_A", {1, 0, 1, 1}},
    {"F12_not_A",   {1, 1, 0, 0}},
    {"F13_A_imp_B", {1, 1, 0, 1}},
    {"F14_NAND",    {1, 1, 1, 0}},
    {"F15_TRUE",    {1, 1, 1, 1}}
};

int binary_inputs[4][2] = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};

const int ETA = 1;
const int EPOCHS = 1;

// --- 2. Funções Auxiliares ---

int to_bipolar(int binary_val) {
    return (binary_val == 1) ? 1 : -1;
}

int to_binary(int bipolar_val) {
    return (bipolar_val > 0) ? 1 : 0;
}

int predict(int weights[3], int inputs_bipolar_with_bias[3]) {
    int net = 0;
    for (int i = 0; i < 3; i++) {
        net += weights[i] * inputs_bipolar_with_bias[i];
    }
    return (net > 0) ? 1 : -1;
}

// --- 3. Função Principal ---

int main() {
    // Arrays para guardar o sumário final
    char results_summary[16][40];
    
    // *** NOVA MODIFICAÇÃO: Matriz para guardar todos os pesos finais ***
    int final_weights_summary[16][3]; 

    printf("Iniciando Treinamento e Teste de Funcoes Logicas com Regra de Hebb\n");
    printf("====================================================================================================\n");

    for (int f = 0; f < 16; f++) {
        printf("\n\nTREINANDO: %s\n", all_16_functions[f].name);

        int weights[3] = {0, 0, 0}; // w = [w0 (bias), w1 (A), w2 (B)]

        // --- 3a. Tabela de Treinamento (Pesos mudando a cada "caso") ---
        printf("\n--- 1. Tabela de Treinamento (Pesos mudando a cada caso) ---\n");
        printf("%-15s | %-19s | %-15s | %-19s | %-19s\n", 
               "Padrão (A,B)", "Entrada [x0,x1,x2]", "Alvo (t_bipolar)", "Δw [w0,w1,w2]", "W_final [w0,w1,w2]");
        printf("----------------+---------------------+-----------------+---------------------+---------------------\n");
        
        for (int e = 0; e < EPOCHS; e++) {
            for (int i = 0; i < 4; i++) { // Loop pelos 4 padrões de entrada
                int A = binary_inputs[i][0];
                int B = binary_inputs[i][1];
                int target_binary = all_16_functions[f].outputs[i];

                int x1_bipolar = to_bipolar(A);
                int x2_bipolar = to_bipolar(B);
                int target_bipolar = to_bipolar(target_binary);

                int x_with_bias[3] = {1, x1_bipolar, x2_bipolar};
                int delta_weights[3] = {0, 0, 0};

                // Aplica a Regra de Hebb: Δw = η * x * t
                for (int j = 0; j < 3; j++) {
                    delta_weights[j] = ETA * x_with_bias[j] * target_bipolar;
                }

                // Atualiza os pesos
                for (int j = 0; j < 3; j++) {
                    weights[j] += delta_weights[j];
                }

                // Imprime a linha da tabela de treinamento
                char pattern_str[10]; sprintf(pattern_str, "(%d,%d)", A, B);
                char x_str[20]; sprintf(x_str, "[1, %2d, %2d]", x1_bipolar, x2_bipolar);
                char dw_str[20]; sprintf(dw_str, "[%2d, %2d, %2d]", delta_weights[0], delta_weights[1], delta_weights[2]);
                char w_str[20]; sprintf(w_str, "[%2d, %2d, %2d]", weights[0], weights[1], weights[2]);
                
                printf("%-15s | %-19s | %-15d | %-19s | %-19s\n", 
                       pattern_str, x_str, target_bipolar, dw_str, w_str);
            }
        }
        
        // *** NOVA MODIFICAÇÃO: Guarda os pesos finais no sumário ***
        for (int j = 0; j < 3; j++) {
            final_weights_summary[f][j] = weights[j];
        }


        // --- 3b. Tabela de Testes (Usando os pesos finais) ---
        printf("\n--- 2. Tabela de Testes (com pesos finais fixos) ---\n");
        printf("Pesos Finais usados: w0=%d, w1=%d, w2=%d\n", weights[0], weights[1], weights[2]);
        
        int num_errors = 0;
        printf("\nEntrada (A,B) | Esperado (t) | Previsto (y) | Status\n");
        printf("----------------+----------------+--------------+----------\n");

        for (int i = 0; i < 4; i++) {
            int A = binary_inputs[i][0];
            int B = binary_inputs[i][1];
            int target_binary = all_16_functions[f].outputs[i];
            int x_with_bias[3] = {1, to_bipolar(A), to_bipolar(B)};
            
            int prediction_bipolar = predict(weights, x_with_bias);
            int prediction_binary = to_binary(prediction_bipolar);

            char status[10] = "";
            if (prediction_binary != target_binary) {
                num_errors++;
                strcpy(status, "<- ERRO");
            }
            
            printf("   (%d, %d)     |      %d       |      %d       | %s\n", 
                   A, B, target_binary, prediction_binary, status);
        }

        // --- 3c. Veredito Final ---
        if (num_errors > 0) {
            printf("\n[VEREDITO: FALHA]\n");
            printf("**ERRO: A funcao %s nao e linearmente separavel.**\n", all_16_functions[f].name);
            strcpy(results_summary[f], "FALHA (Nao Linearmente Separavel)");
        } else {
            printf("\n[VEREDITO: SUCESSO]\n");
            strcpy(results_summary[f], "SUCESSO");
        }
        printf("====================================================================================================\n");
    }

    // --- 4. Sumários Finais ---
    
    // Sumário de Resultados (Sucesso/Falha)
    printf("\n\n========================================================\n");
    printf("SUMARIO DE RESULTADOS (SUCESSO/FALHA)\n");
    printf("========================================================\n");
    
    for (int f = 0; f < 16; f++) {
        printf("%-15s -> %s\n", all_16_functions[f].name, results_summary[f]);
    }

    // *** Pesos Finais ***
    printf("\n\n========================================================\n");
    printf("SUMARIO (MATRIZ) DE PESOS FINAIS\n");
    printf("========================================================\n");
    printf("%-15s | %-20s\n", "Função", "Vetor de Pesos [w0, w1, w2]");
    printf("-----------------+-----------------------\n");
    
    for (int f = 0; f < 16; f++) {
        char w_final_str[30];
        sprintf(w_final_str, "[%2d, %2d, %2d]", 
                final_weights_summary[f][0], // w0 (bias)
                final_weights_summary[f][1], // w1 (A)
                final_weights_summary[f][2]  // w2 (B)
               );
        printf("%-15s | %-20s\n", all_16_functions[f].name, w_final_str);
    }

    return 0;
}
