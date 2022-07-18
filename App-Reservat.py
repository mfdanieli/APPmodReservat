# -*- coding: utf-8 -*-
"""

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

#%% Reservoir model


def concentracao(CARGA):#, taxa_Cin, taxa_Qin, taxa_Qout):
    # dados
    df = pd.read_csv('historico_gbmunhoz.txt',sep='\t')
    df_dbo = pd.read_csv('4_PO4_sintetica_porto_vitoria.txt',sep='\t',header=None,names=['PO4'])
    A = df['AREA (KM2)'].apply(lambda x: x*10**6)  # area de km2 para m2
    V = df['VOLUME (M3)']
    Qout = df['Defluência (m³/s)']#*taxa_Qout  # VOU ASSUMIR QUE A VAZÃO QUE SAI DO RESERVATÓRIO É SÓ A DEFLUENCIA - CONFIRMAR 
    Qin = df['Afluência (m³/s)']#*taxa_Qin 
        
    #CARGA = 429063.712*0.2/1000  # kg/ano to t/yr  
    
    # coef modelo
    vel = 0.000000     # em m/s
    k = 0.04/86400     # em 1/s

    # parâmetros da simulacao
    dt = 86400   # 1 dia em segundos
    n = 731      # dois anos de simulação = 731 dias 

    # cond contorno
    Cout = np.zeros(n)
    Cout[0] = df_dbo.PO4[0]/1000      # COND INICIAL kg/m3
    Cin = df_dbo.PO4/1000 #*taxa_Cin/1000    # kg/m3

    WW=(CARGA*1000)/(86400*365)  # kg/s
    
    # Sol numerica
    carga_permis = []
    carga_reserv = []
    perc_remover = []

    for i in range(n-1):
        Cout[i + 1] = (Cout[i] + (dt/V[i + 1])*(Qin[i + 1]*Cin[i + 1]) + float(WW)*dt/V[i]) / (1 + (dt/V[i + 1])*(Qout[i + 1] + k*V[i + 1] + vel*2*A[i + 1] + (V[i + 1] - V[i])/dt))                                                            
 
    Q95 = np.percentile((Qout), 25)  # 
    carga_permis = (0.03/1000)*(Q95.mean())  # CONFIMRAR kg/s ; tinha pego (Qin[:-1]-Qout[:-1]), mas faz mais sentido a Q_95 do rio > confirmar se peguei o valor certo na linha acima   
    carga_reserv = (Cout[i + 1] )*(Qout) # Qin-Qout ou Qout?
    perc_remover = 100*(1-carga_permis/carga_reserv)
    perc_remover[perc_remover < 0] = 0

    
    return Cout*1000, carga_permis, carga_reserv, perc_remover



#%% Creating the app

# Title
# st.write("""
# APP Teste \n
# Modelo 0D Reservatorio Foz do Areia     
# """)

# Cabeçalho
#st.subheader('')

st.header('Modelo 0D Reservatorio Foz do Areia  ')

with st.expander("How to use"):
    st.write("""
             In the box at left you can select\n
             1. The percent of change of original flow entering the reservoir\n
             
    """)


# dados dos usuários com a função
def get_user_data():
    # taxa_Qin = st.sidebar.slider('Q_Input (choose % of original)', 1.0, 2.0)
    # taxa_Qout = st.sidebar.slider('Q_Output (choose % of original)', 1.0, 2.0)
    CARGA = st.sidebar.slider('Load_Input (t/yr)', 0.0, 1000.0, 0.0)
    # taxa_Cin = st.sidebar.slider('Conc_Input (choose % of original)', 0.1, 2.0, 1.0)
  
    
    # um dicionário recebe as informações acima
    user_data = {'Load_Input': CARGA
                 # 'Conc_Input': taxa_Cin,
                 # 'Q_Input': taxa_Qin,
                 # 'Q_Output': taxa_Qout
                 }
    
    #features = pd.DataFrame(user_data,index=[0])
  
    return CARGA#, taxa_Cin, taxa_Qin, taxa_Qout

user_input_variables = get_user_data()


# Previsao
# prediction, carga_permis, carga_reserv, perc_remover = concentracao(user_input_variables[0],user_input_variables[1],user_input_variables[2],user_input_variables[3])
prediction, carga_permis, carga_reserv, perc_remover = concentracao(user_input_variables)

graf = st.line_chart(prediction)
st.write(prediction)


# # curva permanencia

conc_org = pd.Series(prediction).sort_values(ascending=True) 

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

st.subheader('Carga a remover (media 2 anos)')
st.write(perc_remover.mean())
#st.bar_chart(perc_remover)

# custos remocao 
# custos = 551 euros/kg reduzido
# 

st.write(carga_permis)
st.write(carga_reserv)

#st.subheader('Previsão: ')
#st.write(prediction)


