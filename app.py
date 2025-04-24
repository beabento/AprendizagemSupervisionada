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

st.set_page_config("Probabilidade de inadimplência", "💸")

st.title("Probabilidade de inadimplência")

st.markdown(f"""
Esta Aplicação faz uso de modelos de ML para prever a probabilidade de um aluno se tornar inadimplente.
O modelo usado neste APP foi selecionado por meio de uma validação cruzada, tendo um F1
de {round(modelo['f1']*100)}%. E o Modelo com maior precisáo foi o {modelo['metodo']}.
""")


