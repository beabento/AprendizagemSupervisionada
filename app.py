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

def criar_mapa_formulario_modelo(valores_formulario, valores_modelo):
    """
    Cria um dicionário de mapeamento entre os valores do formulário e os valores do modelo.

    Exemplo:
    'Não' -> 0
    'Sim' -> 1
    """
    mapa = {}
    
    # Verifica se o número de valores do formulário é igual ao número de valores no modelo
    if len(valores_formulario) != len(valores_modelo):
        raise ValueError("O número de valores do formulário deve ser igual ao número de valores no modelo.")

    # Cria o mapeamento entre os valores do formulário e do modelo
    for val_form, val_modelo in zip(valores_formulario, valores_modelo):
        mapa[val_form] = val_modelo

    return mapa

def processar_variavel(nome_variavel, valor_input, mapa, valores_modelo):
    # Aplica o mapeamento do nome amigável para o nome usado no modelo
    valor_mapeado = mapa.get(valor_input, valor_input)

    # Cria dicionário de dummies no formato 'VARIAVEL_VALOR': [0 ou 1]
    dummies = {
        f'{nome_variavel}_{valor}': [1 if valor_mapeado == valor else 0]
        for valor in valores_modelo
    }

    return valor_mapeado, dummies
def processar_multiplas_variaveis(variaveis_info):
    """
    Processa várias variáveis e gera o input_dict com as dummies.
    
    Args:
        variaveis_info (list of dicts): Lista de dicionários com as informações necessárias para processar cada variável.
        
    Returns:
        dict: Dicionário com todas as dummies mapeadas e valores convertidos.
    """
    input_dict = {}

    for variavel in variaveis_info:
        nome_variavel = variavel['nome_variavel']
        valor_input = variavel['valor_input']
        mapa = variavel['mapa']
        valores_modelo = variavel['valores_modelo']
        
        # Processa a variável e cria as dummies
        _, dummies = processar_variavel(
            nome_variavel=nome_variavel,
            valor_input=valor_input,
            mapa=mapa,
            valores_modelo=valores_modelo
        )
        
        # Adiciona as dummies ao input_dict
        input_dict.update(dummies)
    
    return input_dict

# Formulário:

with st.form('user_input_form'):
    genero = st.selectbox('Gênero:', ['Feminino', 'Masculino', 'Indeterminado'])

    pgto_anuidade_2022 = st.selectbox('Pagou a anuidade de 2022?', ['Sim','Não'])

    curso_em_risco_original = st.slider('CURSO EM RISCO (0-5):', 0, 5, 0)

    num_disciplinas_original = st.slider('NUMERO DE DISCIPLINAS MATRICULADAS (0-6):', 0, 6, 0)

    departamento = st.selectbox('DEPARTAMENTO:', ['LIMA', 'CALLAO', 'AMAZONAS', 'ICA', 'AREQUIPA', 'SAN MARTIN',
                                                'JUNIN', 'LA LIBERTAD', 'HUANUCO', 'AYACUCHO', 'ANCASH', 'PASCO',
                                                'CUSCO', 'LAMBAYEQUE', 'HUANCAVELICA', 'PIURA', 'CAJAMARCA',
                                                'APURIMAC', 'PUNO', 'UCAYALI', 'MADRE DE DIOS', 'LORETO', 'TACNA',
                                                'MOQUEGUA', 'TUMBES'])

   
    deficiencia = st.selectbox('DEFICIENCIA:', ['Não', 'Sim'])

    modalidade_ensino = st.selectbox('MODALIDADE DE ENSINO:', ['Presencial', 'Virtual', 'Remoto'])

    turno = st.selectbox('TURNO/HORARIO:', ['Manhã', 'Tarde', 'Noite', 'Misto'])

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
    # Variáveis que já vieram dummy desde o início e que não tiveram novas colunas feitas
    # São essas: PGTO_ANUIDADE_2022 E DEFICIENCIA
    ja_dummy_modelo = [0, 1]
    ja_dummy_form = ['Não', 'Sim']
    ja_dummy_map = criar_mapa_formulario_modelo(ja_dummy_form, ja_dummy_modelo)

    genero_modelo = ['M', 'H', 'I']
    genero_form = ['Feminino', 'Masculino', 'Indeterminado']
    genero_map = criar_mapa_formulario_modelo(genero_form,genero_modelo)

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
        
    departamento_form = ['SAN MARTIN',
                        'LA LIBERTAD',
                        'MADRE DE DIOS',
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

    departamento_map = criar_mapa_formulario_modelo(departamento_form, departamentos_modelo)

    modalidade_ensino_modelo = ['Presencial', 'Virtual', 'Remoto']
    modalidade_ensino_map = criar_mapa_formulario_modelo(modalidade_ensino_modelo, modalidade_ensino_modelo)

    matricula_modelo = ['Novo', 'Reincorporado', 'Reinscrito']
    matricula_map = criar_mapa_formulario_modelo(matricula_modelo, matricula_modelo)

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


    # FACULDADE
    faculdade_modelo = ['Engenharia_Sistemas_Eletrica',
                        'Direito_Ciencias_Politicas',
                        'Administracao_Negocios',
                        'Ciencias_Comunicacao',
                        'Engenharia_Industrial_Mecanica',
                        'Humanas_Ciencias_Sociais',
                        'Contabilidade',
                        'Saude']

    faculdade_form = ['Engenharia Sistemas Eletrica',
                    'Direito Ciencias Politicas',
                    'Administracao Negocios',
                    'Ciencias Comunicacao',
                    'Engenharia Industrial Mecanica',
                    'Humanas Ciencias Sociais',
                    'Contabilidade',
                    'Saude']

    faculdade_map = criar_mapa_formulario_modelo(faculdade_modelo, faculdade_form)


    # FAIXA_ETARIA
    faixa_etaria_modelo = ['Menor_que_18', 'De_19_a_20', 'De_21_a_23', 'De_24_a_29', 'Maior_que_30']
    faixa_etaria_form = ['Menor que 18', '19-20', '21-23', '24-29', 'Maior que 30']
    faixa_etaria_map= criar_mapa_formulario_modelo(faixa_etaria_modelo, faixa_etaria_form)

    # BOLSAS_DESCONTO
    bolsas_desconto_modelo = ['Sem_Beneficio',
                            'Convenios',
                            'Bolsa_Socioecon_Especial',
                            'Bolsa_Socioeconomica',
                            'Bolsa_MadreDeDios',
                            'Bolsa_Alto_Potencial',
                            'Bolsa_Talento']

    bolsas_desconto_form = ['Sem Benefício',
                            'Convenios',
                            'Bolsa Socioeconômica Especial',
                            'Bolsa Socioeconomica',
                            'Bolsa Madre De Dios',
                            'Bolsa Alto Potencial',
                            'Bolsa Talento']

    bolsas_desconto_map = criar_mapa_formulario_modelo(bolsas_desconto_modelo, bolsas_desconto_form)
    # Informações das variáveis
    variaveis_dummy_info = [
        {
            'nome_variavel': 'GENERO',
            'valor_input': genero,
            'mapa': genero_map,
            'valores_modelo': genero_modelo
        },
        {
            'nome_variavel': 'DEPARTAMENTO',
            'valor_input': departamento,
            'mapa': departamento_map,
            'valores_modelo': departamentos_modelo
        },
        {
            'nome_variavel': 'MODALIDADE_ENSINO',
            'valor_input': modalidade_ensino,
            'mapa': modalidade_ensino_map,
            'valores_modelo': modalidade_ensino_modelo
        },
        {
            'nome_variavel': 'TURNO',
            'valor_input': turno,
            'mapa': turno_map,
            'valores_modelo': turno_modelo
        },
        {
            'nome_variavel': 'MATRICULA',
            'valor_input': matricula,
            'mapa': matricula_map,
            'valores_modelo': matricula_modelo
        },
        {
            'nome_variavel': 'CLASSIFICACAO',
            'valor_input': classificacao,
            'mapa': classificacao_map,
            'valores_modelo': classificacao_modelo
        },
        {
            'nome_variavel': 'CAMPUS',
            'valor_input': campus,
            'mapa': campus_map,
            'valores_modelo': campus_modelo
        },
        {
            'nome_variavel': 'FACULDADE',
            'valor_input': faculdade,
            'mapa': faculdade_map,
            'valores_modelo': faculdade_modelo
        },
        {
            'nome_variavel': 'FAIXA_ETARIA',
            'valor_input': faixa_etaria,
            'mapa': faixa_etaria_map,
            'valores_modelo': faixa_etaria_modelo
        },
        {
            'nome_variavel': 'BOLSAS_DESCONTO',
            'valor_input': bolsas_desconto,
            'mapa': bolsas_desconto_map,
            'valores_modelo': bolsas_desconto_modelo
        }
        ]
    input_dict = processar_multiplas_variaveis(variaveis_dummy_info)
    # DEFICIENCIA
    deficiencia = 1 if deficiencia == 'Sim' else 0

    # ANUIDADE
    pgto_anuidade_2022 = 1 if pgto_anuidade_2022 == 'Sim' else 0

    input_dict.update({'PGTO_ANUIDADE_2022' : [pgto_anuidade_2022],
                    'DEFICIENCIA': [deficiencia]})
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

    # Adiciona depois:
    input_dict.update({'CURSO_EM_RISCO': curso_em_risco})

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


    input_dict.update({'NUMERO_DISCIPLINAS_MATRICULADAS': numero_disciplinas_matriculadas})
    input_dict
    # Atualizando dicionário para retirar as variáveis de categoria base das dummies:

    modelo_vars = ['PGTO_ANUIDADE_2022', 'DEFICIENCIA',
                    'NUMERO_DISCIPLINAS_MATRICULADAS', 'CURSO_EM_RISCO',
                    'MATRICULA_Reincorporado', 'MATRICULA_Reinscrito', 'GENERO_I',
                    'GENERO_M', 'DEPARTAMENTO_ANCASH', 'DEPARTAMENTO_APURIMAC',
                    'DEPARTAMENTO_AREQUIPA', 'DEPARTAMENTO_AYACUCHO',
                    'DEPARTAMENTO_CAJAMARCA', 'DEPARTAMENTO_CALLAO', 'DEPARTAMENTO_CUSCO',
                    'DEPARTAMENTO_HUANCAVELICA', 'DEPARTAMENTO_HUANUCO', 'DEPARTAMENTO_ICA',
                    'DEPARTAMENTO_JUNIN', 'DEPARTAMENTO_LAMBAYEQUE',
                    'DEPARTAMENTO_LA_LIBERTAD', 'DEPARTAMENTO_LIMA', 'DEPARTAMENTO_LORETO',
                    'DEPARTAMENTO_MADRE_DE_DIOS', 'DEPARTAMENTO_MOQUEGUA',
                    'DEPARTAMENTO_PASCO', 'DEPARTAMENTO_PIURA', 'DEPARTAMENTO_PUNO',
                    'DEPARTAMENTO_SAN_MARTIN', 'DEPARTAMENTO_TACNA', 'DEPARTAMENTO_TUMBES',
                    'DEPARTAMENTO_UCAYALI', 'CLASSIFICACAO_Graduacao',
                    'CLASSIFICACAO_Graduacao_Semipresencial_50_50',
                    'CLASSIFICACAO_Graduacao_Semipresencial_80_20',
                    'CLASSIFICACAO_Graduacao_Virtual', 'CAMPUS_Beca_18', 'CAMPUS_Chiclayo',
                    'CAMPUS_Chimbote', 'CAMPUS_Huancayo', 'CAMPUS_Ica',
                    'CAMPUS_Lima_Centro', 'CAMPUS_Lima_Este', 'CAMPUS_Lima_Norte',
                    'CAMPUS_Lima_SJL', 'CAMPUS_Lima_Sur', 'CAMPUS_Piura', 'CAMPUS_Trujillo',
                    'CAMPUS_Virtual', 'FACULDADE_Ciencias_Comunicacao',
                    'FACULDADE_Contabilidade', 'FACULDADE_Direito_Ciencias_Politicas',
                    'FACULDADE_Engenharia_Industrial_Mecanica',
                    'FACULDADE_Engenharia_Sistemas_Eletrica',
                    'FACULDADE_Humanas_Ciencias_Sociais', 'FACULDADE_Saude', 'TURNO_Misto',
                    'TURNO_Noite', 'TURNO_Tarde', 'BOLSAS_DESCONTO_Bolsa_MadreDeDios',
                    'BOLSAS_DESCONTO_Bolsa_Socioecon_Especial',
                    'BOLSAS_DESCONTO_Bolsa_Socioeconomica', 'BOLSAS_DESCONTO_Bolsa_Talento',
                    'BOLSAS_DESCONTO_Convenios', 'BOLSAS_DESCONTO_Sem_Beneficio',
                    'FAIXA_ETARIA_De_21_a_23', 'FAIXA_ETARIA_De_24_a_29',
                    'FAIXA_ETARIA_Maior_que_30', 'FAIXA_ETARIA_Menor_que_18']

    novo_dict = {}

    for i in modelo_vars:
        novo_dict[i] = input_dict[i]
    input_dict = novo_dict 

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


