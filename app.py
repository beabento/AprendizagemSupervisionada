import pandas as pd
import streamlit as st
import pickle
import numpy
from sklearn.preprocessing import LabelEncoder
import os



# Importado o melhor modelo do arquivo pkl:
modelo = pickle.load(open('data/modelo_final.pkl', 'rb'))
params = modelo['resultados']

# Criando função de classificação:
def classificar(df):
    "Função de predição/classificação"
    previsoes = params.predict(df)
    return previsoes, params.predict_proba(df)



# Configuração da página:

st.set_page_config("PrevInad: previna a inadimplência", "💸")

st.title("PrevInad")

st.markdown(f"""
PrevInad é uma aplicação que prevê a probabilidade de um aluno se tornar inadimplente, 
através do uso de modelos de Aprendizado de Máquina (Machine Learning).

O modelo usado neste APP foi selecionado por meio de uma validação cruzada, tendo um F1
de {round(modelo['f1']*100)}%. E o Modelo com maior precisão foi o {modelo['metodo']}.
""")

st.header('Preencha os dados do aluno:')



# Criando função de encoders:

def load_encoders(columns, load_dir='data'):
    """
    Carrega múltiplos encoders previamente salvos como arquivos .pkl.

    Parâmetros:
    - columns: lista de nomes de colunas (ex: ['DEPARTAMENTO', 'PROVINCIA']).
    - load_dir: diretório onde os arquivos .pkl estão salvos.

    Retorna:
    - Um dicionário com nome_da_coluna → encoder
    """
    encoders = {}

    for col in columns:
        filename = f"{col.lower().replace('/', '_').replace(' ', '_')}_encoder.pkl"
        #filename = f"{col.lower()}_encoder.pkl"
        file_path = os.path.join(load_dir, filename)

        with open(file_path, 'rb') as f:
            encoders[col] = pickle.load(f)

    return encoders

columns = ['MATRICULA', 'PAGAMENTO DE ANUIDADE MARÇO 2022', 'PAGAMENTO DE ANUIDADE MARÇO 2023',
        'GENERO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'CLASSIFICACAO', 'CAMPUS',
        'FACULDADE', 'PROGRAMA/CURSO', 'TURNO/HORARIO', 'BOLSAS DE DESCONTO',
        'MODALIDADE DE ENSINO', 'FAIXA ETARIA', 'DEFICIENCIA',
        'CURSO EM RISCO']

encoders = load_encoders(columns, load_dir='data')

def encode_multiple_inputs(input_dict, encoders):
    all_encoded = []

    for col, value in input_dict.items():
        # Se a variável não tem encoder, usa valor diretamente
        if col not in encoders:
            if isinstance(value, list):
                value = value[0]
            temp_df = pd.DataFrame({col: [value]})
            all_encoded.append(temp_df)
            continue

        # Caso contrário, aplica encoder
        encoder = encoders[col]
        if isinstance(value, list):
            value = value[0]
        temp_df = pd.DataFrame({col: [value]})
        encoded_array = encoder.transform(temp_df[[col]])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out([col])
        )
        all_encoded.append(encoded_df)

    return pd.concat(all_encoded, axis=1)

def processar_variavel(nome_variavel, valor_input, mapa, valores_modelo):
    # Aplica o mapeamento do nome amigável para o nome usado no modelo
    valor_mapeado = mapa.get(valor_input, valor_input)

    # Cria dicionário de dummies no formato 'VARIAVEL_VALOR': [0 ou 1]
    dummies = {
        f'{nome_variavel}_{valor}': [1 if valor_mapeado == valor else 0]
        for valor in valores_modelo
    }

    return valor_mapeado, dummies

# Formulário:

with st.form('user_input_form'):
    genero = st.selectbox('Gênero:', ['Feminino', 'Masculino', 'Indeterminado'])

    pgto_anuidade_2022 = st.selectbox('Pagou a anuidade de 2022?', ['Sim','Não'])

    curso_em_risco_original = st.slider('CURSO EM RISCO (0-5):', 0, 5, 0)
    # Normalização para o intervalo [0. , 0.4, 0.2, 0.6, 1. , 0.8]
    # Vamos mapear os valores originais para os normalizados.
    # Assumindo uma correspondência linear aproximada, mas a ordem dos normalizados não é estritamente crescente.
    # Uma maneira mais robusta seria ter um dicionário de mapeamento se a relação não for linear.
    mapeamento_curso_risco = {
        0: 0.0,
        1: 0.4,
        2: 0.2,
        3: 0.6,
        4: 1.0,
        5: 0.8
    }
    curso_em_risco = mapeamento_curso_risco[curso_em_risco_original]

    num_disciplinas_original = st.slider('NUMERO DE DISCIPLINAS MATRICULADAS (0-6):', 0, 6, 0)
    # Normalização para o intervalo [0.        , 0.5       , 0.16666667, 0.33333333, 0.66666667, 0.83333333, 1.        ]
    # Novamente, assumindo uma correspondência por índice, já que a relação não é estritamente linear.
    # Uma maneira mais robusta seria ter um dicionário de mapeamento.
    mapeamento_num_disciplinas = {
        0: 0.0,
        1: 0.5,
        2: 0.16666667,
        3: 0.33333333,
        4: 0.66666667,
        5: 0.83333333,
        6: 1.0
    }
    numero_disciplinas_matriculadas = mapeamento_num_disciplinas[num_disciplinas_original]

    # Agora, 'curso_em_risco_normalizado' e 'num_disciplinas_normalizado' contêm os valores
    # normalizados que você pode usar para alimentar seu modelo de machine learning
    # dentro do DataFrame 'df'. Por exemplo:
    # df['CURSO EM RISCO'] = [curso_em_risco_normalizado]
    # df['NUMERO DE DISCIPLINAS MATRICULADAS'] = [num_disciplinas_normalizado]

    departamento = st.selectbox('DEPARTAMENTO:', ['LIMA', 'CALLAO', 'AMAZONAS', 'ICA', 'AREQUIPA', 'SAN MARTIN',
                                                'JUNIN', 'LA LIBERTAD', 'HUANUCO', 'AYACUCHO', 'ANCASH', 'PASCO',
                                                'CUSCO', 'LAMBAYEQUE', 'HUANCAVELICA', 'PIURA', 'CAJAMARCA',
                                                'APURIMAC', 'PUNO', 'UCAYALI', 'MADRE DE DIOS', 'LORETO', 'TACNA',
                                                'MOQUEGUA', 'TUMBES'])

   
    deficiencia = st.selectbox('DEFICIENCIA:', ['Não', 'Sim'])

    modalidade_ensino = st.selectbox('MODALIDADE DE ENSINO:', ['Presencial', 'Virtual', 'Remoto'])

    turno_horario = st.selectbox('TURNO/HORARIO:', ['Manhã', 'Tarde', 'Noite', 'Misto'])

    matricula = st.selectbox('MATRICULA:', ['Novo', 'Reincorporado', 'Reinscrito'])

    classificacao = st.selectbox('CLASSIFICACAO:', ['Graduação',
                                                    'Graduação Semipresencial(50-50)',
                                                    'Graduação Semipresencial (80-20)',
                                                    'Curso para Trabalhadores',
                                                    'Graduação Virtual'])

    campus = st.selectbox('CAMPUS:', ['Lima Centro',
                                        'Lima Norte',
                                        'Lima SJL',
                                        'Lima Este',
                                        'Lima Sur',
                                        'Arequipa',
                                        'Chiclayo',
                                        'Piura',
                                        'Chimbote',
                                        'Huancayo',
                                        'Ica',
                                        'Beca 18',
                                        'Trujillo',
                                        'Virtual'])

    faculdade = st.selectbox('FACULDADE:', ['Engenharia Sistemas Eletrica',
                                            'Direito Ciencias Politicas',
                                            'Administracao Negocios',
                                            'Ciencias Comunicacao',
                                            'Engenharia Industrial Mecanica',
                                            'Humanas Ciencias Sociais',
                                            'Contabilidade',
                                            'Saude'])

    faixa_etaria = st.selectbox('FAIXA ETARIA:', ['Menor que 18', '19-20', '21-23', '24-29', 'Maior que 30'])

    bolsas_desconto = st.selectbox('BOLSAS DE DESCONTO:', ['Sem Benefício',
                                                            'Convenios',
                                                            'Bolsa Socioeconômica Especial',
                                                            'Bolsa Socioeconomica',
                                                            'Bolsa Madre De Dios',
                                                            'Bolsa Alto Potencial',
                                                            'Bolsa Talento'])

    # Every form must have a submit button
    submitted = st.form_submit_button("Enviar")

# Ao submeter
if submitted:
    # GENERO
    genero_m = 1 if genero == 'Masculino' else 0
    genero_u = 1 if genero == 'Indeterminado' else 0

    # DEFICIENCIA
    deficiencia_modelo = [0, 1]
    deficiencia_form = ['Não', 'Sim']
    deficiencia_map = criar_mapa_formulario_modelo(deficiencia_form, deficiencia_modelo)

    # TURNO
    turno_modelo = ['Manha', 'Tarde', 'Noite', 'Misto']
    turno_form = ['Manhã', 'Tarde', 'Noite', 'Misto']
    turno_map = criar_mapa_formulario_modelo(turno_form,turno_modelo)


    # DEPARTAMENTO
    departamentos_modelo = ['SAN_MARTIN',
                            'LA_LIBERTAD',
                            'MADRE_DE_DIOS',
                            'LIMA',
                            'CALLAO',
                            'AMAZONAS',
                            'ICA',
                            'AREQUIPA',
                            'JUNIN',
                            'HUANUCO',
                            'AYACUCHO',
                            'ANCASH',
                            'PASCO',
                            'CUSCO',
                            'LAMBAYEQUE',
                            'HUANCAVELICA',
                            'PIURA',
                            'CAJAMARCA',
                            'APURIMAC',
                            'PUNO',
                            'UCAYALI',
                            'LORETO',
                            'TACNA',
                            'MOQUEGUA',
                            'TUMBES']
    
    departamento_map = {
        "SAN MARTIN": "SAN_MARTIN",
        "LA LIBERTAD": "LA_LIBERTAD",
        "MADRE DE DIOS": "MADRE_DE_DIOS"
    }

# CLASSIFICACAO:
    classificacao_modelo = [
        'Graduacao',
        'Graduacao_Semipresencial_50_50',
        'Graduacao_Semipresencial_80_20',
        'Curso_para_Trabalhadores',
        'Graduacao_Virtual'
    ]
    
    classificacao_map = {
        'Graduação':'Graduacao',
        'Graduação Semipresencial(50-50)':'Graduacao_Semipresencial_50_50',
        'Graduação Semipresencial (80-20)':'Graduacao_Semipresencial_80_20',
        'Curso para Trabalhadores':'Curso_para_Trabalhadores',
        'Graduação Virtual':'Graduacao_Virtual'
    }

# CAMPUS
    campus_modelo = [
        'Lima_Centro',
        'Lima_Norte',
        'Lima_SJL',
        'Lima_Este',
        'Lima_Sur',
        'Arequipa',
        'Chiclayo',
        'Piura',
        'Chimbote',
        'Huancayo',
        'Ica',
        'Beca_18',
        'Trujillo',
        'Virtual']
    
    campus_map = {
        'Lima Centro': 'Lima_Centro',
        'Lima Norte': 'Lima_Norte',
        'Lima SJL': 'Lima_SJL',
        'Lima Este': 'Lima_Este',
        'Lima Sur': 'Lima_Sur',
        'Arequipa': 'Arequipa',
        'Chiclayo': 'Chiclayo',
        'Piura': 'Piura',
        'Chimbote': 'Chimbote',
        'Huancayo': 'Huancayo',
        'Ica': 'Ica',
        'Beca 18': 'Beca_18',
        'Trujillo': 'Trujillo',
        'Virtual': 'Virtual'
    }


    # Variáveis separadas por tipo:
    dummies = [
    'PGTO_ANUIDADE_2022', 'MATRICULA', 'GENERO', 'DEPARTAMENTO', 'CLASSIFICACAO', 
    'CAMPUS', 'FACULDADE', 'TURNO', 'BOLSAS_DESCONTO', 'FAIXA_ETARIA']

    numericas = ['NUMERO_DISCIPLINAS_MATRICULADAS', 'CURSO_EM_RISCO']

    # OUTRAS VARIÁVEIS:

    
    # Cria as dummies para DEPARTAMENTO


    # Cria dicionário de entrada final
    input_dict = {
        'GENERO_M': [genero_m],
        'GENERO_U': [genero_u],
        **departamento_dummies
        # Adicione outras variáveis aqui, se houver
    }

    # Aplica o encode, se necessário
    encoded_df = encode_multiple_inputs(input_dict, encoders)

    # Faz a predição com o modelo de regressão logística
    pred = params.predict(encoded_df)[0]
    prob = params.predict_proba(encoded_df)[0][1]  # probabilidade da classe 1, se for binário

    # Exibe o resultado
    st.markdown("### Resultado da Predição")
    st.write(f"**Classe prevista:** {pred}")
    st.write(f"**Probabilidade (risco):** {prob:.2%}")


# if submitted:
#     # Codifica os inputs
#     encoded_df = encode_multiple_inputs(input_dict, encoders)

#     # Faz a predição com o modelo de regressão logística
#     pred = params.predict(encoded_df)[0]
#     prob = params.predict_proba(encoded_df)[0][1]  # probabilidade da classe 1, se for binário

#     # Exibe o resultado
#     st.markdown("### Resultado da Predição")
#     st.write(f"**Classe prevista:** {pred}")
#     st.write(f"**Probabilidade (risco):** {prob:.2%}")


