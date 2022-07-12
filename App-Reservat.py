# -*- coding: utf-8 -*-
"""
Here it is possible to input new values for risk factors 
and predict if the pacient has diabetes in real-time
"""
#!/usr/bin/env python
# coding: utf-8

#%% Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import xlrd
import altair as alt

#%% Import the dataset
# df = pd.read_csv("diabetes.csv")


#%% Reservoir model


def concentracao(CARGA, taxa_Cin, taxa_Qin, taxa_Qout):
    #taxa_Cin = 1
    # dados
    df = pd.read_csv('historico_gbmunhoz.txt',sep='\t')
    df_dbo = pd.read_csv('4_PO4_sintetica_porto_vitoria.txt',sep='\t',header=None,names=['PO4'])
    A = df['AREA (KM2)'].apply(lambda x: x*10**6)  # area de km2 para m2
    V = df['VOLUME (M3)']
    Qout = df['Defluência (m³/s)']*taxa_Qout  # VOU ASSUMIR QUE A VAZÃO QUE SAI DO RESERVATÓRIO É SÓ A DEFLUENCIA - CONFIRMAR COM CRIS
    Qin = df['Afluência (m³/s)']*taxa_Qin 
        
    #CARGA = 429063.712*0.2/1000  # kg/ano to t/yr  
    
    # coef modelo
    vel = 0.000000 # em m/s
    k = 0.04/86400     # em 1/s

    # parâmetros da simulacao
    dt = 86400   # 1 dia em segundos
    n = 731      # dois anos de simulação = 731 dias 

    # cond contorno
    Cout = np.zeros(n)
    Cout[0] = df_dbo.PO4[0]/1000        # COND INICIAL - TROCAR DEPOIS ; kg/m3
    Cin = df_dbo.PO4*taxa_Cin/1000    # kg/m3

    WW=(CARGA*1000)/(86400*365)  # kg/s
    
    # Sol numerica
    carga_permis = []
    carga_reserv = []
    perc_remover = []
    # carga_permis = np.zeros(n)
    # carga_permis = np.zeros(n)
    # carga_reserv = np.zeros(n)
    # carga_reserv[0] = 0
    # perc_remover = np.zeros(n)
    # perc_remover[0] = 0.
    for i in range(n-1):
        Cout[i + 1] = (Cout[i] + (dt/V[i + 1])*(Qin[i + 1]*Cin[i + 1]) + float(WW)*dt/V[i]) / (1 + (dt/V[i + 1])*(Qout[i + 1] + k*V[i + 1] + vel*2*A[i + 1] + (V[i + 1] - V[i])/dt))                                                            
        #conc_out.append(Cout)
    #return conc_out*1000
    Q95 = np.percentile(Qin, 25)  # perc
    carga_permis = (0.03/1000)*(Q95)  # kg/s ; tinha pego (Qin[:-1]-Qout[:-1]), mas faz mais sentido a Q_95 do rio > confirmar se peguei o valor certo na linha acima
    carga_reserv = (Cout[i + 1] )*(Qin-Qout)
    perc_remover = 100*(1-carga_permis/carga_reserv)
    perc_remover[perc_remover < 0] = 0
    # print (perc_remover)
    return Cout*1000, carga_permis, carga_reserv, perc_remover



#%% Creating the app

# Title
st.write("""
APP Teste \n
Modelo 0D Reservatorio Foz do Areia     
""")

# Cabeçalho
st.subheader('Informações dos dados')

# Nome do usuário
# user_input = st.sidebar.text_input('Digite seu nome')

# st.write('Paciente: ', user_input)

# dados dos usuários com a função
def get_user_data():
    taxa_Qin = st.sidebar.slider('Q_Input (choose % of original)', 0.1, 2.0, 1.0)
    taxa_Qout = st.sidebar.slider('Q_Output (choose % of original)', 0.1, 2.0, 1.0)
    CARGA = st.sidebar.slider('Load_Input (t/yr)', 0.0, 1000.0, 0.0)
    taxa_Cin = st.sidebar.slider('Conc_Input (choose % of original)', 0.1, 2.0, 1.0)
    # A = st.sidebar.slider('Area', 0.0, 900.0, 30.0)
    # vel = st.sidebar.slider('Veloc', 0.01, 70.0, 15.0)
    # V = st.sidebar.slider('Volume', 0.0, 3.0, 0.0)
    # k = st.sidebar.slider('Kinetics', 15, 100, 21)
    
    # um dicionário recebe as informações acima
    # user_data = {'VazInput': Qin,
    #              'VazOutput': Qout,
    #             'ConcInput': Cin,
    #             'LoadInput': WW,
    #             'Area': A,
    #             'Veloc': vel,
    #             'Volume': V,
    #             'Kinetics': k
    #              }
    user_data = {'Load_Input': CARGA,
                 'Conc_Input': taxa_Cin,
                 'Q_Input': taxa_Qin,
                 'Q_Output': taxa_Qout
                 }
    
    #features = pd.DataFrame(user_data,index=[0])
  
    return CARGA, taxa_Cin, taxa_Qin, taxa_Qout

user_input_variables = get_user_data()

# gráfico
# graf = st.bar_chart(user_input_variables)
    
st.subheader('Carga definida pelo usuario')
st.write(user_input_variables[0])
#st.write(taxa_Cin)
st.write(type(user_input_variables))


# Previsao
## prediction = concentracao(user_input_variables_standard)
prediction, carga_permis, carga_reserv, perc_remover = concentracao(user_input_variables[0],user_input_variables[1],user_input_variables[2],user_input_variables[3])


graf = st.line_chart(prediction)
st.write(prediction)


# # curva permanencia

conc_org = pd.Series(prediction).sort_values(ascending=True) 
#prediction.sort_values()
# graf = st.line_chart(conc_org)
exceedence = np.arange(1.,len(conc_org)+1) / len(conc_org)
data = {'Conc':conc_org, 'Freq':exceedence*100, 'Clase': np.ones(len(conc_org))*0.03}
data_f = pd.DataFrame(data)
st.write(data_f)
#graf2 = st.line_chart(data_f)

chart = alt.Chart(data_f).mark_line().encode(
    x='Freq',
    y='Conc'
)            

classe = alt.Chart(data_f).mark_line(opacity=0.6,color='red').encode(
    x='Freq',
    y='Clase'
)   

chart + classe

excedencia = sum(i > 0.03 for i in prediction)
st.subheader('Número de vezes em que classe 2 é excedida')
st.write(excedencia)

st.subheader('Carga a remover')
st.write(carga_perms)
st.bar_chart(perc_remover)

#st.subheader('Previsão: ')
#st.write(prediction)


