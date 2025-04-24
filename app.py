import pandas as pd
import streamlit as st
import pickle

# Importado o melhor modelo do arquivo pkl:
modelo = pickle.load(open('data/modelo_final.pkl', 'rb'))
params = modelo['resultados']

# Criando função de classificação:
def classificar(df):
    "Função de predição/classificação"
    previsoes = params.predict(df)
    return previsoes, params.predict_proba(df)

st.set_page_config("PrevInad: previna a inadimplência", "💸")

st.title("PrevInad")

st.markdown(f"""
PrevInad é uma aplicação que prevê a probabilidade de um aluno se tornar inadimplente, 
através do uso de modelos de Aprendizado de Máquina (Machine Learning).

O modelo usado neste APP foi selecionado por meio de uma validação cruzada, tendo um F1
de {round(modelo['f1']*100)}%. E o Modelo com maior precisão foi o {modelo['metodo']}.
""")

st.header('Preencha os dados do aluno:')

sexo = st.selectbox('Sexo:', ['Feminino', 'Masculino'])
idade = st.number_input('Idade ', value=40, placeholder='Digite sua idade')
pag_2022 = st.selectbox('Pagou a anuidade de 2022?', ['Sim','Não'])
st.markdown(f"""... e assim em diante: temos que colocar todas as variáveis e opções do nosso modelo.
            Ver o exemplo que o professor deu sobre o titanic para referencia.
            Pode ser que escolhemos variáveis demais e que nem todas sejam relevantes.
            O ideal é a gente enxugar o nosso modelo e gerar de novo o pkl, com no máximo 5 ou 7 variáveis.
            Tem que ver no codigo projeto_excola_peru.ipynb, no mesmo repositório do github.
            Simbora!""")
