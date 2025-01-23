
# **Previsão de Renda**

## **Descrição do Projeto**
Este projeto tem como objetivo prever a renda mensal de clientes com base em características demográficas, econômicas e sociais. O modelo desenvolvido auxilia instituições financeiras a otimizar estratégias de crédito, reduzir riscos financeiros e personalizar ofertas aos clientes.

## **Metodologia Utilizada**
O projeto foi desenvolvido seguindo as etapas da metodologia **CRISP-DM**:
1. **Entendimento do Negócio**: Compreender o objetivo do modelo e sua aplicação prática.
2. **Entendimento dos Dados**: Analisar e explorar o conjunto de dados para garantir sua qualidade.
3. **Preparação dos Dados**: Limpeza, transformação e seleção de variáveis relevantes.
4. **Modelagem**: Construção e avaliação de diferentes modelos preditivos.
5. **Avaliação**: Comparação de métricas de desempenho e escolha do melhor modelo.
6. **Implantação**: Criação de um aplicativo interativo para previsão da renda em tempo real.

## **Tecnologias e Bibliotecas Utilizadas**
- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Seaborn e Matplotlib
- Streamlit (para criação do aplicativo)
- ydata_profiling (para análise exploratória)

## **Conjunto de Dados**
- **Fonte**: Conjunto de dados fornecido pela EBAC para fins educacionais (previsao_de_renda.csv)
- **Descrição**: O dataset contém informações sobre clientes, como idade, tempo de emprego, nível educacional, estado civil, tipo de renda, entre outros.
- **Dicionário de Dados**:
  - `sexo`: Gênero do cliente (F ou M).
  - `idade`: Idade do cliente em anos.
  - `tempo_emprego`: Tempo de emprego em anos.
  - `tipo_renda`: Categoria da fonte de renda (Assalariado, Empresário, etc.).
  - `educacao`: Nível educacional (Secundário, Superior completo, etc.).
  - `renda`: Renda mensal do cliente (variável alvo).
    
## **Profiling Report: [Clique aqui para acessar a análise de renda](./renda_analysis.html)

## **Modelos Testados**
Os seguintes modelos foram avaliados:
1. **Regressão Linear (Statsmodels)**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Árvore de Decisão**

### **Resultados**
| Modelo                | MSE            | MAE            | R²   |
|-----------------------|----------------|----------------|-------|
| Regressão Linear      | 10,703,865.28 | 2,437.34       | 0.27  |
| Ridge Regression      | 10,703,669.71 | 2,437.30       | 0.27  |
| Lasso Regression      | 10,703,780.03 | 2,434.58       | 0.27  |
| **Árvore de Decisão** | **10,439,283.71** | **2,373.35** | **0.29** |

A **Árvore de Decisão** apresentou o melhor desempenho em todas as métricas avaliadas.

## **Aplicação do Projeto**
Foi desenvolvido um aplicativo interativo com **Streamlit** que permite:
- Inserir informações sobre clientes, como idade, escolaridade e estado civil.
- Obter uma previsão imediata da renda mensal do cliente.

### **Demonstração**
[streamlit-app-2025-01-22-23-01-49.mov.webm](https://github.com/user-attachments/assets/437fe854-9a50-48d4-871a-f5a8a4d13229)


## **Como Executar o Projeto**
1. Clone este repositório:
   ```bash
   git clone https://github.com/adrianajikal/previsao-de-renda.git
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute o aplicativo:
   ```bash
   streamlit run app.py
   ```

## **Contribuições**
Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

## **Contato**
- Autor: Adriana Jikal
- LinkedIn: [www.linkedin.com/in/adrianajikal]
- Email: adrijikal@gmail.com
---

