import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Previsão de renda",
     page_icon="💰",
     layout="wide",
)

# tabs configuration

tab1, tab2 = st.tabs(["PREVISÃO DE RENDA",
                      "ANÁLISE EXPLORATÓRIA DA PREVISÃO DE RENDA"])


with tab1:
     # abrindo o arquivo
    df = pd.read_csv('previsao_de_renda.csv')
     # definindo uma mascara para outliers
    df = df[(df['renda'] <= 20000) & (df['renda'] >= 500)] 
    df = df.drop(columns=['Unnamed: 0', 'id_cliente', 'data_ref'])
    df = df.dropna() 
    df = pd.get_dummies(df, columns=['sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia'], drop_first=True)

     # rodando o modelo Arvore de Decisao
    X = df.drop(['renda'], axis=1)
    y = df['renda']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
    tree = DecisionTreeRegressor(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    y_pred_tree = tree.predict(X_test)
    
    def predict_income(sexo, posse_de_veiculo, posse_de_imovel, qtd_filhos, educacao, estado_civil, tipo_residencia, idade, tempo_emprego, qt_pessoas_residencia):
        input_data = {
            'sexo': sexo,
            'posse_de_veiculo': posse_de_veiculo,
            'posse_de_imovel': posse_de_imovel,
            'qtd_filhos': qtd_filhos,
            'educacao': educacao,
            'estado_civil': estado_civil,
            'tipo_residencia': tipo_residencia,
            'idade': idade,
            'tempo_emprego': tempo_emprego,
            'qt_pessoas_residencia': qt_pessoas_residencia,
            
        }

        # Crie um DataFrame a partir dos dados de entrada
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

        # Faça a previsão usando seu modelo
        prediction = tree.predict(input_df)

        return prediction[0]

    # Interface do aplicativo usando Streamlit

    st.title("Aplicativo de Predição de Renda")

    # Widgets para inserção de valores
    sexo = st.radio("Sexo", ("M", "F"))
    posse_de_veiculo = st.checkbox("Possui Veículo?")
    posse_de_imovel = st.checkbox("Possui Imóvel?")
    qtd_filhos = st.number_input("Quantidade de Filhos", min_value=0, max_value=14, value=0)
    educacao = st.selectbox("Educação", ("Secundário", "Superior completo", "Superior incompleto", "Primário", "Pós Graduação"))
    estado_civil = st.selectbox("Estado Civil", ("Solteiro", "Casado", "Separado", "União", "Viúvo"))
    tipo_residencia = st.selectbox("Tipo de Residência", ("Casa", "Com os pais", "Aluguel", "Estúdio", "Governamental", "Comunitário"))
    idade = st.slider("Idade", min_value=18, max_value=100, value=30)
    tempo_emprego = st.slider("Tempo de Emprego", min_value=0, max_value=42, value=10)
    qt_pessoas_residencia = st.slider("Quantidade de adultos na Residência", min_value=1, max_value=15, value=2)

    # Botão para fazer a predição
    if st.button("**Prever**"):
        # Transformar valores de entrada em formato apropriado
        posse_de_veiculo = True if posse_de_veiculo else False
        posse_de_imovel = True if posse_de_imovel else False
        
        # Fazer a previsão usando a função
        prediction = predict_income(sexo, posse_de_veiculo, posse_de_imovel, qtd_filhos, educacao, estado_civil, tipo_residencia, idade, tempo_emprego, qt_pessoas_residencia)

        # Formatando para resultado em português
        formatted_prediction = f"R$ {prediction:.2f}"

        # Mostrando resultado
        st.write(f"A previsão de renda é: {formatted_prediction}")
    

with tab2:
    

    st.write("<h1 style='text-align: center; color: #696969;'>ANÁLISE EXPLORATÓRIA DA PREVISÃO DE RENDA</h1>", unsafe_allow_html=True)


    renda = pd.read_csv('./input/previsao_de_renda.csv')

    #plots
    fig, ax = plt.subplots(8,1,figsize=(20,80))
    plt.rc('legend', fontsize=12)

    renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
    ax[0].legend(loc='upper right')
    ax[0].set_title("Renda por Posse de Imovel")

    st.write('## Gráficos ao longo do tempo')

    sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].legend(loc='upper right')
    ax[1].set_title("Renda ao Longo do Tempo por Posse de Imóvel")

    sns.lineplot(x='data_ref', y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
    ax[2].tick_params(axis='x', rotation=45)
    ax[2].legend(loc='upper right')
    ax[2].set_title("Renda ao Longo do Tempo por Posse de Veiculo")

    sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
    ax[3].tick_params(axis='x', rotation=45)
    ax[3].legend(loc='upper right')
    ax[3].set_title("Renda ao Longo do Tempo por Quantidade de Filhos")

    sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
    ax[4].tick_params(axis='x', rotation=45)
    ax[4].legend(loc='upper right')
    ax[4].set_title("Renda ao Longo do Tempo por Tipo de Renda")

    sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
    ax[5].tick_params(axis='x', rotation=45)
    ax[5].legend(loc='upper right')
    ax[5].set_title("Renda ao Longo do Tempo por Educação")

    sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
    ax[6].tick_params(axis='x', rotation=45)
    ax[6].legend(loc='upper right')
    ax[6].set_title("Renda ao Longo do Tempo por Estado Civil")

    sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
    ax[7].tick_params(axis='x', rotation=45)
    ax[7].legend(loc='upper right')
    ax[7].set_title("Renda ao Longo do Tempo por Tipo de Residência")

    plt.subplots_adjust(hspace=0.8)  # Espaçamento vertical entre os gráficos

    sns.despine()
    st.pyplot(plt)

    st.write('## Gráficos bivariada')
    fig, ax = plt.subplots(7,1,figsize=(20,80))

    sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0], palette='coolwarm', legend=False)
    ax[0].set_title("Renda por Posse de Imóvel")

    sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1], palette='coolwarm', legend=False)
    ax[1].set_title("Renda por Posse de Veiculo")

    sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2], palette='coolwarm', legend=False)
    ax[2].set_title("Renda por Quantidade de Filhos")

    sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3], palette='coolwarm', legend=False)
    ax[3].set_title("Renda por Tipo de Renda")

    sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4], palette='coolwarm', legend=False)
    ax[4].set_title("Renda por Nível de Educação")

    sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5], palette='coolwarm', legend=False)
    ax[5].set_title("Renda por Estado Civil")

    sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6], palette='coolwarm', legend=False)
    ax[6].set_title("Renda por Tipo de Residência")

    plt.subplots_adjust(hspace=0.6)

    sns.despine()
    st.pyplot(plt)