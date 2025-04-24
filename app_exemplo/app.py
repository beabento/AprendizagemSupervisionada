import pandas as pd
import pickle
import streamlit as st
# 1. Carregar o best model
modelo = pickle.load(open('data/modelo_final.pkl', 'rb'))
params = modelo['resultados']

# Função de classificação
def make_prediction(df):
    "Função de predição/classificação"
    previsoes = params.predict(df)
    return previsoes, params.predict_proba(df)

st.set_page_config("Previsão de sobrevivência Titanic", "🚢")

st.title('Classificador de Sobrevivências em Acidentes - Titanic')
st.markdown(f"""
Esta Aplicação faz uso de modelos de ML para prever casos de óbitos e sobrevivências em desastres naufrágicos.
O modelo usado neste APP foi selecionado por meio de uma validação cruzada, tendo uma acurácia
de {round(modelo['acuracia']*100)}%. E o Modelo com maior precisáo foi o {modelo['metodo']}.
""")

st.header('Formulário de entrada de dados')

sexo = st.selectbox('Sexo:', ['Feminino', 'Masculino'])
idade = st.number_input('Idade ', value=40, placeholder='Digite sua idade')

age0 = modelo['escala'][0]['age'][0]
age1 = modelo['escala'][0]['age'][1]

df = pd.DataFrame({
    'age': [ (idade - age0)/(age1-age0) ],
    'sibsp': [0.361169], 'parch': [0], 'fare': [0.412503],
    'pclass_2': [False], 'pclass_3': [False], 
    'sex_male': [sexo=='Masculino'],
    'embarked_Q': [False], 'embarked_S': [True]
})

df['age'] = df['age'].clip(0,1)

st.subheader('Dados normalizados')
st.table(df)

if st.button('Prever'):
    st.header('Resultados')

    prev, prob = make_prediction(df)
    chart_data = pd.DataFrame(
        {"Probabilidade": prob[0],
         "Classe prevista": ['Morreria', 'Sobreviveria']}
    )

    st.bar_chart(chart_data, x="Classe prevista", y="Probabilidade")

    classe = "Óbito"
    if prev[0]==1:
        classe = 'Sobrevivente'
    st.write('Classe predita:', classe)
    st.write('Probabilidades:')
    for classe, probabilidade in zip(params.classes_, prob[0]):
        st.write(f'{classe}: {probabilidade:.2f}')
        

