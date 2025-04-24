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

genero = st.selectbox('Gênero:', ['Feminino', 'Masculino', 'Indeterminado'])

pag_2022 = st.selectbox('Pagou a anuidade de 2022?', ['Sim','Não'])

departamento = st.selectbox('DEPARTAMENTO:', ['LIMA', 'CALLAO', 'AMAZONAS', 'ICA', 'AREQUIPA', 'SAN MARTIN',
                                            'JUNIN', 'LA LIBERTAD', 'HUANUCO', 'AYACUCHO', 'ANCASH', 'PASCO',
                                            'CUSCO', 'LAMBAYEQUE', 'HUANCAVELICA', 'PIURA', 'CAJAMARCA',
                                            'APURIMAC', 'PUNO', 'UCAYALI', 'MADRE DE DIOS', 'LORETO', 'TACNA',
                                            'MOQUEGUA', 'TUMBES'])

provincia = st.selectbox('PROVINCIA:', ['LIMA', 'CALLAO', 'RODRIGUEZ DE MENDOZA', 'CHINCHA', 'CAYLLOMA',
                                          'SAN MARTIN', 'CANETE', 'YAULI', 'NAZCA', 'HUAURA', 'TRUJILLO',
                                          'HUANCAYO', 'HUANUCO', 'BARRANCA', 'ICA', 'HUAROCHIRI', 'CANTA',
                                          'OYON', 'VICTOR FAJARDO', 'AREQUIPA', 'HUAMANGA', 'SANTA',
                                          'PARINACOCHAS', 'OXAPAMPA', 'CUSCO', 'HUARAL', 'LAMBAYEQUE',
                                          'YAUYOS', 'CHANCHAMAYO', 'CAJATAMBO', 'HUANCAVELICA',
                                          'PAUCAR DEL SARA SARA', 'HUANCABAMBA', 'BAGUA', 'TARMA', 'CHOTA',
                                          'MARISCAL CACERES', 'HUARAZ', 'CHICLAYO', 'CAJAMARCA', 'CUTERVO',
                                          'JUNIN', 'CASMA', 'ANDAHUAYLAS', 'CHINCHEROS', 'PIURA', 'CHEPEN',
                                          'LA CONVENCION', 'PUNO', 'YUNGAY', 'CORONEL PORTILLO',
                                          'SANTA CRUZ', 'CHACHAPOYAS', 'ABANCAY', 'TAMBOPATA', 'MOYOBAMBA',
                                          'PISCO', 'SAN ROMAN', 'COTABAMBAS', 'PATAZ', 'ESPINAR', 'PASCO',
                                          'AYABACA', 'YUNGUYO', 'TALARA', 'SUCRE', 'MAYNAS', 'JAEN', 'JAUJA',
                                          'CARAVELI', 'LUCANAS', 'TACNA', 'CARHUAZ', 'SATIPO', 'HUALLAGA',
                                          'HUANTA', 'ANTA', 'PAITA', 'TOCACHE', 'LEONCIO PRADO', 'HUAYLAS',
                                          'RIOJA', 'SIHUAS', 'CASTROVIRREYNA', 'ALTO AMAZONAS', 'LAMAS',
                                          'HUARI', 'UTCUBAMBA', 'PACASMAYO', 'SAN IGNACIO', 'CAMANA',
                                          'BOLOGNESI', 'SULLANA', 'ATALAYA', 'POMABAMBA', 'ANTONIO RAIMONDI',
                                          'BONGARA', 'LA MAR', 'DOS DE MAYO', 'CHUPACA', 'CARABAYA',
                                          'URUBAMBA', 'PADRE ABAD', 'QUISPICANCHIS', 'CONCEPCION',
                                          'DANIEL ALCIDES CARRION', 'ANGARAES', 'ISLAY', 'TAYACAJA',
                                          'RECUAY', 'CARLOS FERMIN FITZCARRALD', 'CANGALLO', 'ACOBAMBA',
                                          'CASTILLA', 'AZANGARO', 'MARISCAL NIETO', 'ILO', 'CONDESUYOS',
                                          'CANCHIS', 'CHUMBIVILCAS', 'EL COLLAO', 'LA UNION',
                                          'GENERAL SANCHEZ CERRO', 'MELGAR', 'CALCA', 'CANAS', 'HUANCANE',
                                          'LAMPA', 'ANTABAMBA', 'CHUCUITO', 'JORGE BASADRE', 'SANDIA',
                                          'FERRENAFE', 'EL DORADO', 'CONDORCANQUI', 'SAN PABLO',
                                          'BELLAVISTA', 'CELENDIN', 'SAN MIGUEL', 'TUMBES', 'MORROPON',
                                          'CAJABAMBA', 'SECHURA', 'ZARUMILLA', 'HUARMEY', 'VIRU', 'REQUENA',
                                          'ASUNCION', 'MARANON', 'PALLASCA', 'HUAYTARA', 'PALPA', 'AYMARAES',
                                          'ASCOPE', 'MOHO', 'CONTRALMIRANTE VILLAR', 'UCAYALI',
                                          'DATEM DEL MARAÑON', 'TARATA', 'LUYA', 'MANU', 'LORETO',
                                          'CHURCAMPA', 'HUALGAYOC', 'PUERTO INCA'])

distrito = st.selectbox('DISTRITO:', ['BRENA', 'VILLA MARIA DEL TRIUNFO', 'JESUS MARIA', 'ATE',
                                        'SURQUILLO', 'SAN JUAN DE LURIGANCHO', 'EL AGUSTINO', 'CALLAO',
                                        'CARMEN DE LA LEGUA-REYNOSO', 'PUEBLO LIBRE', 'COMAS',
                                        'BELLAVISTA', 'CHORRILLOS', 'LIMA', 'LOS OLIVOS', 'SAN MIGUEL',
                                        'LINCE', 'PUNTA NEGRA', 'SAN MARTIN DE PORRES', 'LONGAR',
                                        'LA VICTORIA', 'SUNAMPE', 'MAJES', 'SANTIAGO DE SURCO', 'TARAPOTO',
                                        'MAGDALENA DEL MAR', 'RIMAC', 'SAN LUIS', 'QUILMANA', 'HUAY HUAY',
                                        'SAN JUAN DE MIRAFLORES', 'MIRAFLORES', 'EL INGENIO', 'HUACHO',
                                        'TRUJILLO', 'PACHACAMAC', 'EL TAMBO', 'CIENEGUILLA', 'LURIN',
                                        'SANTA ANITA', 'VILLA EL SALVADOR', 'HUANUCO', 'LA MOLINA',
                                        'LURIGANCHO', 'INDEPENDENCIA', 'BARRANCA', 'LA TINGUINA',
                                        'SANTA EULALIA', 'MATUCANA', 'CANTA', 'VENTANILLA',
                                        'SAN VICENTE DE CANETE', 'CHINCHA ALTA', 'OYON', 'MALA',
                                        'LEONCIO PRADO', 'HUANCAPI', 'HUAROCHIRI', 'SAN BORJA',
                                        'SAN ANTONIO', 'CARABAYLLO', 'PUENTE PIEDRA', 'LA PERLA', 'ICA',
                                        'PUCUSANA', 'SAN ISIDRO', 'CHACLACAYO', 'JACOBO HUNTER',
                                        'AYACUCHO', 'NUEVO CHIMBOTE', 'LA OROYA', 'CORACORA', 'OXAPAMPA',
                                        'NUEVO IMPERIAL', 'CUSCO', 'HUARAL', 'JAYANCA', 'BARRANCO',
                                        'CARAMPOMA', 'YAUYOS', 'CHANCHAMAYO', 'HUANCAYO', 'IMPERIAL',
                                        'PARCONA', 'CAJATAMBO', 'HUANCAVELICA', 'MARCABAMBA',
                                        'HUANCABAMBA', 'COPALLIN', 'SAN RAMON', 'TARMA', 'ARAMANGO',
                                        'ANCON', 'ALTO SELVA ALEGRE', 'CHANCAY', 'CHILCA', 'LAJAS',
                                        'JUANJUI', 'CHICLAYO', 'MONSEFU', 'SANTIAGO', 'PUEBLO NUEVO',
                                        'CAJAMARCA', 'QUEROCOTILLO', 'SANTA ROSA', 'CERRO COLORADO',
                                        'JUNIN', 'PARAMONGA', 'YURA', 'TIABAYA', 'CASMA', 'ANDAHUAYLAS',
                                        'ANCO HUALLO', 'PIURA', 'YANAHUARA', 'SUPE', 'CHEPEN', 'SAYAN',
                                        'KIMBIRI', 'MARCONA', 'AMARILIS', 'PICHACANI', 'NAZCA',
                                        'SANTA CRUZ DE FLORES', 'AREQUIPA', 'YUNGAY', 'CALLERIA', 'CAYMA',
                                        'LUNAHUANA', 'SANTA CRUZ', 'SANTA', 'CHACHAPOYAS', 'ABANCAY',
                                        'TAMBOPATA', 'MOYOBAMBA', 'HUMAY', 'ALTO LARAN', 'WANCHAQ',
                                        'LA PUNTA', 'SALAS', 'LAMBAYEQUE', 'JULIACA', 'CHALLHUAHUACHO',
                                        'SAN SEBASTIAN', 'CASTILLA', 'CARHUAMAYO', 'MARIANO MELGAR',
                                        'PATAZ', 'ESPINAR', 'LA ESPERANZA', 'CHIMBOTE', 'SAN BARTOLO',
                                        'CHINCHA BAJA', 'PALLANCHACRA', 'CARMEN ALTO', 'GROCIO PRADO',
                                        'SANTA MARIA', 'CALANGO', 'CERRO AZUL', 'MORROPE', 'SOCABAYA',
                                        'AYABACA', 'SAN JUAN BAUTISTA', 'SAN ANDRES', 'YUNGUYO',
                                        'TALAVERA', 'PARINAS', 'PACAYCASA', 'QUEROBAMBA', 'IQUITOS',
                                        'SANTO DOMINGO DE LOS OLLEROS', 'EL ALTO', 'JAUJA', 'CARAVELI',
                                        'MORALES', 'PUQUIO', 'HUARAZ', 'TACNA', 'OCOBAMBA', 'CARHUAZ',
                                        'PISCO', 'ASIA', 'MAZAMARI', 'LOS AQUIJES', 'SATIPO', 'CATACAOS',
                                        'HUALMAY', 'SAPOSOA', 'MANANTAY', 'CURAHUASI', 'HUANTA', 'ANTA',
                                        'HUANCAN', 'PAITA', 'TOCACHE', 'RUPA RUPA', 'LA JOYA',
                                        'SAN JERONIMO', 'PUNO', 'CARAZ', 'PULLO', 'RIOJA', 'SIHUAS',
                                        'CHINCHAO', 'SICAYA', 'JESUS NAZARENO', 'AURAHUA',
                                        'JOSE LEONARDO ORTIZ', 'YURIMAGUAS', 'PINTO RECODO', 'PICHANAQUI',
                                        'YARINACOCHA', 'JAEN', 'HUAURA', 'MOROCOCHA', 'PANGOA',
                                        'SUPE PUERTO', 'JOSE LUIS BUSTAMANTE Y RIVERO', 'HUARI', 'MASMA',
                                        'SORITOR', 'BAGUA GRANDE', 'SANTA ROSA DE QUIVES', 'YANACANCHA',
                                        'OLMOS', 'SAN PEDRO DE LLOC', 'IGUAIN', 'TABACONAS', 'AUCALLAMA',
                                        'CHOTA', 'CUTERVO', 'JOSE MARIA QUIMPER', 'COLQUIOC', 'NEPEÑA',
                                        'PILCOMAYO', 'STA MARIA DEL MAR', 'SAN MARCOS', 'SULLANA', 'YURUA',
                                        'ACARI', 'VEGUETA', 'POMABAMBA', 'SAN MIGUEL DE EL FAIQUE',
                                        'CHINGAS', 'AWAJUN', 'PAROBAMBA', 'YAMON', 'JUMBILLA', 'TAMBO',
                                        'LA UNION', 'BELEN', 'CHONGOS BAJO', 'VISTA ALEGRE', 'MORO',
                                        'AYAPATA', 'MARAS', 'PADRE ABAD', 'SAN NICOLAS', 'MANAS', 'RAGASH',
                                        'SAN BARTOLOME', 'CHAPARRA', 'RAHUAPAMPA', 'ACZO',
                                        'VICTOR LARCO HERRERA', 'URCOS', 'ANTIOQUIA', 'HUASAHUASI',
                                        'NUEVO PROGRESO', 'CONCEPCION', 'CHINCHEROS', 'YAUCA', 'CHUPACA',
                                        'CONAYCA', 'CUTURAPI', 'CHACAYAN', 'CAJARURO', 'PERENE',
                                        'SAN FCO DE ASIS DE YARUSYACAN', 'CHONTABAMBA', 'VITIS',
                                        'STA MARIA DE CHICMO', 'SANTA CRUZ DE COCACHACRA', 'RICARDO PALMA',
                                        'SURCO', 'SAN MATEO', 'YANAHUANCA', 'SAN CLEMENTE', 'PAUCARTAMBO',
                                        'CAMPOVERDE', 'LIRCAY', 'MOLLENDO', 'TINYAHUARCO',
                                        'DANIEL HERNANDEZ', 'SAN MATEO DE OTAO', 'SIMON BOLIVAR',
                                        'HUARIACA', 'CHAUPIMARCA', 'HUACHON', 'VILLA RICA', 'QUICHES',
                                        'YAULI', 'HUANCHACO', 'TAPUC', 'SAN LUIS DE SHUARO', 'CHICLA',
                                        'PIMENTEL', 'PUNTA HERMOSA', 'LLACLLIN', 'HUAMBO', 'CHIARA',
                                        'CHALA', 'JAZAN', 'PAUSA', 'PAUCARPATA', 'CARMEN SALCEDO',
                                        'COAYLLO', 'CANGALLO', 'PICHARI', 'APONGO', 'POMACOCHA',
                                        'UCHUMAYO', 'HUANCARQUI', 'ASILLO', 'MOQUEGUA', 'CHARACATO',
                                        'SACHACA', 'ILO', 'COPANI', 'CAMANA', 'COTABAMBAS', 'OCONA',
                                        'CIUDAD NUEVA', 'COR GREG ALBARR L.ALF UGARTE',
                                        'STA RITA DE SIGUAS', 'CHUQUIBAMBA', 'NICOLAS DE PIEROLA',
                                        'MARIANO N VALCARCEL', 'SICUANI', 'ATICO', 'ISLAY',
                                        'SAMUEL PASTOR', 'PACOCHA', 'COCACHACRA', 'LLUSCO', 'SABANDIA',
                                        'HUANUHUANU', 'MOLLEBAYA', 'ALTO DE LA ALIANZA', 'AZANGARO',
                                        'VIRACO', 'COTAHUASI', 'YANQUE', 'MARISCAL CACERES', 'YUNGA',
                                        'APLAO', 'OMATE', 'CAYLLOMA', 'MEJIA', 'YARABAMBA', 'CHIGUATA',
                                        'DEAN VALDIVIA', 'ANTAUTA', 'SANTO TOMAS', 'CALCA',
                                        'SANTA ISABEL DE SIGUAS', 'LANGUI', 'BELLA UNION', 'TORATA',
                                        'VELILLE', 'URUBAMBA', 'VITOR', 'TARACO', 'CHIVAY', 'TUTI',
                                        'OCORURO', 'ORCOPAMPA', 'ATIQUIPA', 'SANTA LUCIA', 'YANAQUIHUA',
                                        'AYAVIRI', 'COPORAQUE', 'TAMBOBAMBA', 'LAYO', 'PALLPATA',
                                        'JUAN ESPINOZA MEDRANO', 'PUNTA DE BOMBON', 'COATA', 'ILAVE',
                                        'QUEQUENA', 'SAMEGUA', 'QUINOTA', 'URACA', 'SAN CRISTOBAL',
                                        'UMACHIRI', 'POMATA', 'OLLACHEA', 'HAQUIRA', 'LLALLI', 'MACUSANI',
                                        'ITE', 'SAN GABAN', 'ILABAYA', 'DESAGUADERO', 'TUPAC AMARU',
                                        'JULI', 'ZEPITA', 'QUILCA', 'LAMPA', 'COLQUEMARCA', 'SAN ANTON',
                                        'ACORA', 'TAPAY', 'MANAZO', 'PUCARA', 'SANDIA', 'SAMAN',
                                        'RIO GRANDE', 'INAMBARI', 'CABANILLAS', 'TISCO', 'CHAMACA',
                                        'ICHUNA', 'POLOBAYA', 'CAHUACHO', 'ECHARATE', 'NUNOA', 'MUNANI',
                                        'SANTA ANA', 'CORANI', 'ARAPA', 'SAN JUAN DE SIGUAS', 'AJOYANI',
                                        'LAS PIEDRAS', 'HUANCANE', 'MOCHUMI', 'POMALCA', 'SAN JOSE',
                                        'OYOTUN', 'FLORIDA', 'FERRENAFE', 'REQUE', 'MOTUPE', 'ETEN PUERTO',
                                        'SANA', 'PITIPO', 'CAYALTI', 'BAGUA', 'TUCUME', 'ILLIMO',
                                        'SAN JOSE DE SISA', 'PACANGA', 'GUADALUPE', 'SOCOTA', 'PATAPO',
                                        'TACABAMBA', 'HUAMBOS', 'ETEN', 'PICSI', 'CHONGOYAPE', 'TUMAN',
                                        'PACORA', 'PUCALA', 'CHOROS', 'INCAHUASI', 'NIEVA', 'LAGUNAS',
                                        'SAN PABLO', 'LAS LOMAS', 'SAN IGNACIO', 'SAN JUAN DE LICUPIS',
                                        'CELENDIN', 'CHALAMARCA', 'HUARMACA', 'LA COIPA',
                                        'NUEVA CAJAMARCA', 'IMAZA', 'OMIA', 'PACASMAYO', 'TONGOD',
                                        'CANARIS', 'TUMBES', 'COLASAY', 'QUEROCOTO', 'PULAN', 'NAMBALLE',
                                        'TAMBO GRANDE', 'BUENOS AIRES', 'SANTO DOMINGO DE LA CAPILLA',
                                        'EL MILAGRO', 'SUYO', 'CATACHE', 'ELIAS SOPLIN', 'JAMALCA',
                                        'SAN FELIPE', 'NINABAMBA', 'PUNCHANA', 'CAJABAMBA',
                                        'SANTA CATALINA DE MOSSA', 'SAN JOSE DE LOURDES',
                                        'VEINTISEIS DE OCTUBRE', 'IGNACIO ESCUDERO', 'LA ARENA', 'AMOTAPE',
                                        'CHULUCANAS', 'SECHURA', 'CURA MORI', 'LA BREA', 'MANCORA',
                                        'QUERECOTILLO', 'MARCAVELICA', 'LALAQUIZ', 'AGUAS VERDES',
                                        'MORROPON', 'SALITRAL', 'SONDORILLO', 'PAIMAS', 'ZARUMILLA',
                                        'PUEBLO NUEVO DE COLAN', 'LA HUACA', 'EL TALLAN', 'LOS ORGANOS',
                                        'VICE', 'FRIAS', 'CANCHAQUE', 'PACAIPAMPA', 'MIGUEL CHECA',
                                        'LANCONES', 'CHALACO', 'BELLAVISTA DE LA UNION', 'PAPAYAL',
                                        'RINCONADA LLICUAR', 'POMAHUACA', 'VICHAYAL', 'SANTO DOMINGO',
                                        'PAMPAS DE HOSPITAL', 'TAYABAMBA', 'COISHCO', 'HUARMEY', 'CHAO',
                                        'ALTO TAPICHE', 'GUADALUPITO', 'YAUTAN', 'YUNGAR', 'CHACAS',
                                        'HUACRACHUCO', 'BUENA VISTA ALTA', 'RECUAY', 'MOCHE', 'SAMANCO',
                                        'PAMPAS', 'COCHABAMBA', 'CONCHUCOS', 'YANAMA', 'CACERES DEL PERU',
                                        'SUBTANJALLA', 'SAPALLANGA', 'SANO', 'SAN JERONIMO DE TUNAN',
                                        'HUALHUAS', 'HUAYUCACHI', 'HUANCA HUANCA', 'SAN AGUSTIN', 'PACCHA',
                                        'HUAMANCACA CHICO', 'AHUAC', 'STA ROSA DE SACCO', 'COCHAS',
                                        'STA ROSA DE OCOPA', 'TINTAY PUNCU', 'PALCAZU', 'SAUSA',
                                        'LLOCLLAPAMPA', 'MATAHUASI', 'MITO', 'CHICCHE',
                                        'TRES DE DICIEMBRE', 'HUAYLLAY', 'CCOCHACCASA', 'PARCO', 'VIQUES',
                                        'HUACHAC', 'NUEVE DE JULIO', 'CHUSCHI', 'ULCUMAYO', 'SAN LORENZO',
                                        'RIO TAMBO', 'ACOBAMBA', 'ATAURA', 'QUILCAS', 'ACOLLA', 'ORCOTUNA',
                                        'COLCABAMBA', 'PAUCARA', 'SAN ANTONIO DE CUSICANCHA', 'TAPO',
                                        'PANCAN', 'PAZOS', 'MCAL CASTILLA', 'SAN JUAN DE ISCOS',
                                        'HUAYTARA', 'LLIPATA', 'TUPAC AMARU INCA',
                                        'SAN JOSE DE LOS MOLINOS', 'PACHACUTEC', 'TATE', 'TAMBO DE MORA',
                                        'PARACAS', 'HUANCANO', 'PALPA', 'OCUCAJE', 'CHALHUANCA', 'CANARIA',
                                        'PUYUSCA', 'EL PORVENIR', 'CASA GRANDE', 'FLORENCIA DE MORA',
                                        'LAREDO', 'TAMBURCO', 'SAN JACINTO', 'MOHO', 'RAIMONDI',
                                        'ZORRITOS', 'AYNA', 'PATIVILCA', 'ATAQUERO', 'LA CRUZ',
                                        'LOS BANOS DEL INCA', 'MARCO', 'PARAS', 'CORRALES', 'COTARUSE',
                                        'CONTAMANA', 'LURICOCHA', 'PUERTO BERMUDEZ', 'TARATA', 'PONTO',
                                        'UCHIZA', 'SAYLLA', 'SAN BUENAVENTURA', 'OCALLI', 'YANATILE',
                                        'SANTA TERESA', 'SAN PEDRO DE PILLAO', 'MATALAQUE', 'MATAPALO',
                                        'LA BANDA DE SHILCAYO', 'CALETA DE CARQUIN', 'YAUYA', 'LLUTA',
                                        'CHUPA', 'PLATERIA', 'EL CENEPA', 'SAN PEDRO DE PUTINA PUNCO',
                                        'UCO', 'VILCABAMBA', 'EL CARMEN', 'CRUCERO', 'CHIRINOS', 'BERNAL',
                                        'HUEPETUHE', 'EL PRADO', 'HUAMATAMBO', 'INGENIO', 'POCOLLAY',
                                        'JANGAS', 'AHUAYCHA', 'PIRA', 'NAUTA', 'COASA', 'HUARO', 'CHECCA',
                                        'SAN PEDRO DE CORIS', 'SITAJARA', 'CACHACHI', 'MARA', 'PITUMARCA',
                                        'CHIQUIAN', 'HUARANGO', 'BAMBAMARCA', 'PACUCHA', 'TANTARA',
                                        'SHUNQUI', 'NINACACA', 'CASTROVIRREYNA', 'YUYAPICHIS', 'PICHIGUA',
                                        'SAN PEDRO DE LARCAY', 'LA PECA', 'PATAMBUCO', 'ASCENCION',
                                        'REQUENA', 'CALZADA', 'SAN MARTIN'])

deficiencia = st.selectbox('DEFICIENCIA:', ['No', 'Si'])

modalidade_ensino = st.selectbox('MODALIDADE DE ENSINO:', ['Presencial', 'Virtual', 'Remoto'])

turno_horario = st.selectbox('TURNO/HORARIO:', ['MIXTO', 'NOCHE', 'MAÑANA', 'TARDE'])

programa_curso = st.selectbox('PROGRAMA/CURSO:', ['ING. DE REDES Y COMUNICACIONES', 'ING. DE SISTEMAS', 'DERECHO',
                                            'ADMINISTRACION DE NEGOCIOS (50/50)',
                                            'ADM. DE NEGOCIOS INTERNACIO', 'DISEÑO DIGITAL PUBLICITARIO',
                                            'ING. ELECTRONICA', 'ING. DE DISEÑO GRAFICO',
                                            'INGENIERIA INDUSTRIAL (50/50)',
                                            'ING. DE SISTEMAS E INFORMÁTICA (80/20)',
                                            'ING. DE TELECOMUNICACIONES',
                                            'ADM. DE NEGOCIOS Y FINANZAS (50/50)', 'ADM. DE EMPRESAS (80/20)',
                                            'PSICOLOGIA (80/20)', 'INGENIERIA INDUSTRIAL', 'ADM. DE EMPRESAS',
                                            'PSICOLOGIA', 'ING. BIOMEDICA', 'CONTABILIDAD',
                                            'NEGOCIOS INTERNACIONALES (50/50)', 'TERAPIA FÍSICA',
                                            'ING. DE SOFTWARE', 'CONTABILIDAD FINANCIERA (50/50)',
                                            'ING. AUTOMOTRIZ', 'INGENIERÍA EMPRESARIAL', 'ENFERMERÍA',
                                            'ING. INDUSTRIAL', 'INGENIERÍA CIVIL (50/50)',
                                            'CIENCIAS DE LA COMUNICACIÓN', 'ING. ELECTRICA Y DE POTENCIA',
                                            'ARQUITECTURA', 'OBSTETRICIA', 'ING. AERONAUTICA',
                                            'ADM. Y MARKETING', 'ADM. DE NEGOCIOS Y MARKETING (50/50)',
                                            'ADM. HOTELERA Y DE TURISMO', 'DERECHO (80/20)',
                                            'ING. MECATRONICA', 'DERECHO (50/50)',
                                            'ING. DE SISTEMAS E INFORMÁTICA (50/50)',
                                            'COMUNICACIÓN Y PUBLICIDAD', 'ING. CIVIL',
                                            'ADM. DE NEGOCIOS Y MARKETING', 'ADM. BANCA Y FINANZAS',
                                            'ING. MARITIMA', 'ING. MECANICA', 'ADMINISTRACION Y NEGOCIOS',
                                            'ING. ELECTROMECANICA', 'INGENIERIA DE SISTEMAS E INFOR',
                                            'NEGOCIOS INTERNACIONALES', 'CONTABILIDAD FINANCIERA',
                                            'CONTABILIDAD (80/20)', 'ING. ECONOMICA Y EMPRESARIAL',
                                            'INGENIERIA CIVIL', 'ING. EN SEGURIDAD LAB. Y AMB.',
                                            'INGENIERÍA EMPRESARIAL (50/50)', 'ING. INDUSTRIAL (80/20)',
                                            'ADM. DE NEGOCIOS Y FINANZAS', 'INGENIERÍA EMPRESARIAL CGT',
                                            'NUTRICIÓN', 'PSICOLOGÍA (50/50)', 'ING. TEXTIL Y DE CONFECCIONES',
                                            'INGENIERÍA DE SOFTWARE', 'ING. DE DISEÑO COMPUTACIONAL',
                                            'ADMINISTRACION Y MARKETING', 'ING. DE SEGURIDAD Y AUDIT. INF',
                                            'INGENIERIA DE SISTEMAS', 'INGENIERÍA CIVIL',
                                            'ADMINISTRACION DE NEGOCIOS', 'ING. DE SEGUR. INDUST. Y MINER',
                                            'INGENIERÍA DE MINAS', 'ING. DE SEGUR. INDUST. Y MINER (50/50)',
                                            'ING. DE SISTEMAS E INFORMÁTICA', 'INGENIERIA INDUSTRIAL (B18)',
                                            'INGENIERIA MECANICA (B18)', 'INGENIERÍA BIOMÉDICA (B18)',
                                            'INGENIERIA DE TELECOMUNICACION', 'ING. ELECTRICA Y DE POTENC B18',
                                            'ADMINISTRACION HOTELERA  Y DE', 'ADM. DE EMPRESAS (VIRTUAL)',
                                            'DERECHO (VIRTUAL)', 'ING. INDUSTRIAL (VIRTUAL)',
                                            'ING. DE SISTEMAS E INFORMÁTICA (VIRTUAL)', 'PSICOLOGIA (VIRTUAL)',
                                            'CONTABILIDAD (VIRTUAL)'])

matricula = st.selectbox('MATRICULA:', ['Nuevo', 'Reincorporado', 'Reinscrito'])

classificacao = st.selectbox('CLASSIFICACAO:', ['Carreras Pregrado', 'Carreras Pregrado 50-50',
                                                'Carreras Pregrado 80-20', 'Carreras PPE',
                                                'Carreras Pregrado Virtual'])

campus = st.selectbox('CAMPUS:', ['UTP Lima Centro', 'UTP Lima Norte', 'UTP SJL', 'UTP Lima Este',
                                    'UTP Lima Sur', 'UTP Arequipa', 'UTP Chiclayo', 'UTP Piura',
                                    'UTP Chimbote', 'UTP Huancayo', 'UTP Ica', 'UTP Beca 18',
                                    'UTP Trujillo', 'UTP Virtual'])

faculdade = st.selectbox('FACULDADE:', ['Fac. Ing. Sist. Y Elect.', 'Fac. Der. Cienc. Polit. Y RRII',
                                        'Fac. Adm. Y Neg.', 'Fac. Cienc. Com.', 'Fac. Ing. Ind. Y Mec.',
                                        'Fac. Hum y CC Soc', 'Fac. Contabilidad', 'Fac. Salud'])

faixa_etaria = st.selectbox('FAIXA ETARIA:', ['5. >=30', '4. 24-29', '3. 21-23', '2. 19-20', '1. <=18'])

bolsas_desconto = st.selectbox('BOLSAS DE DESCONTO:', ['SIN BENEFICIO', 'CONVENIOS', 'SOCIOECONOMICA ESPECIAL - UTP',
                                                    'SOCIOECONOMICA - UTP', 'MADREDIOSENSE - UTP',
                                                    'BECA ALTO POTENCIAL', 'BECA TALENTO UTP'])


st.markdown(f"""... e assim em diante: temos que colocar todas as variáveis e opções do nosso modelo.
            Ver o exemplo que o professor deu sobre o titanic para referencia.
            Pode ser que escolhemos variáveis demais e que nem todas sejam relevantes.
            O ideal é a gente enxugar o nosso modelo e gerar de novo o pkl, com no máximo 5 ou 7 variáveis.
            Tem que ver no codigo projeto_escola_peru.ipynb, no mesmo repositório do github.
            Simbora!""")
