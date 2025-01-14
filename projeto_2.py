import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Previsão de Renda",
     page_icon="💰",
     layout="wide"
)


st.write("<h1 style='text-align: center; color: #696969;'>ANÁLISE EXPLORATÓRIA DA PREVISÃO DE RENDA</h1>", unsafe_allow_html=True)


renda = pd.read_csv('./input/previsao_de_renda.csv')

#plots
fig, ax = plt.subplots(8,1,figsize=(10,50))
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
ax[7].set_title("Renda ao Longo do Tempo por Tipo de ResidÊncia")

plt.subplots_adjust(hspace=0.8)  # Espaçamento vertical entre os gráficos

sns.despine()
st.pyplot(plt)

st.write('## Gráficos bivariada')
fig, ax = plt.subplots(7,1,figsize=(10,40))
paleta = 'Set1'
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0], palette=paleta)
ax[0].set_title("Renda por Posse de Imóvel")

sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1], palette=paleta)
ax[1].set_title("Renda por Posse de Veiculo")

sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2], palette=paleta)
ax[2].set_title("Renda por Quantidade de Filhos")

sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3], palette=paleta)
ax[3].set_title("Renda por Tipo de Renda")

sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4], palette=paleta)
ax[4].tick_params(axis='x', rotation=45)
ax[4].set_title("Renda por Nível de Educação")

sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5], palette=paleta)
ax[5].set_title("Renda por Estado Civil")

sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6], palette=paleta)
ax[6].tick_params(axis='x', rotation=45)
ax[6].set_title("Renda por Tipo de Residência")

plt.subplots_adjust(hspace=1)

sns.despine()
st.pyplot(plt)





