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
st.markdown(f"""... e assim em diante: temos que colocar todas as variáveis e opções do nosso modelo""")

""" exemplo titanic pra referencia:
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
        
"""

