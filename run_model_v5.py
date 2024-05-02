import os
import numpy as np
from scipy.stats import zscore
import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
import biogeme.messaging as msg
import biogeme.optimization as opt
import biogeme.results as res
from biogeme.expressions import Beta, Variable, bioDraws, log, MonteCarlo, bioNormalCdf
import logging


df = pd.read_csv('norm_df.csv')
database = db.Database('latvars', df)

##################### variables dependientes #####################
SCL_mean=Variable('mean_scl')
SCR_freq=Variable('scr_frequency')
SCL_AMP=Variable('mean_amplitude')
SCL_RT=Variable('mean_risetime')
SCL_MAX=Variable('max_scl')
speed=Variable('speed_kmph')
##################### variables de independientes #####################

############## variables contextuales ##############
pressure=Variable('pressure_10s_log_sum')
CO2=Variable('CO2_10s_log_sum')
temp=Variable('temperature_10s_log_sum')
hum=Variable('humidity_10s_log_sum')
temphum = temp*hum
noise=Variable('ambientNoise_10s_log_sum')
bright=Variable('brightness_10s_log_sum')

############## variables de viaje ##############
distance_to_int=Variable('distance_to_intersection')
pendiente=Variable('binary_slope')
time=Variable('time_from_beginning')
time_of_day=Variable('Horario')

############## variables de infraestructura ##############
# tipo intersecciones
inter_paso_cebra = Variable('interseccion_Cebra')
inter_ceda_paso =Variable('interseccion_Ceda el paso')
inter_pare = Variable('interseccion_Pare')
# tipo de tramo
s_ciclovia =Variable('sin_ciclovia')
park =Variable('parque_o_paseo')
# Ancho pista
ancho =Variable('Ancho [m]')
# Pista bidireccional
bid_si =Variable('Bidireccional_si')
# Sentido de autos
car_way_si =Variable('Sentido autos_si')
# Estacionamiento en vía
park_si =Variable('estacionamiento_si')
# número de pistas
nlanes =Variable('pistas de auto')


############## Variables demográficas ##############
gen =Variable('Genero')
edad =Variable('Edad')
ruta_xp =Variable('Conoce')
bike_xp =Variable('Experiencia años')
bike_freq =Variable('Frecuencia /mes')
estudia =Variable('Estudia')
trabaja = Variable('Trabaja')

Const= Beta('Const', 0, None, None, 0)

# Contextual variables betas
B_pressure = Beta('B_pressure', 0, None, None, 0)
B_CO2 = Beta('B_CO2', 0, None, None, 0)
B_temp = Beta('B_temp', 0, None, None, 0)
B_hum = Beta('B_hum', 0, None, None, 0)
B_temphum = Beta('B_temphum', 0, None, None, 0)
B_noise = Beta('B_noise', 0, None, None, 0)
B_bright = Beta('B_bright', 0, None, None, 0)

# Travel variables betas
B_distance_to_int = Beta('B_distance_to_int', 0, None, None, 0)
B_pendiente = Beta('B_pendiente', 0, None, None, 0)
B_time = Beta('B_time', 0, None, None, 0)
B_time_of_day = Beta('B_time_of_day', 0, None, None, 0)

# Infrastructure variables betas
B_inter_paso_cebra = Beta('inter_paso_cebra', 0, None, None, 0)
B_inter_ceda_paso = Beta('inter_ceda_paso', 0, None, None, 0)
B_inter_pare = Beta('inter_pare', 0, None, None, 0)

B_s_ciclovia =Beta('s_ciclovia', 0, None, None, 0)
B_park =Beta('park', 0, None, None, 0)

B_ancho =Beta('ancho', 0, None, None, 0)

B_bid_si =Beta('bid_si', 0, None, None, 0)

B_car_way_si =Beta('car_way_si', 0, None, None, 0)

B_park_si =Beta('park_si', 0, None, None, 0)

B_nlanes =Beta('nlanes', 0, None, None, 0)

# Demographic variables betas
B_gen = Beta('B_gen', 0, None, None, 0)
B_edad = Beta('B_edad', 0, None, None, 0)
B_ruta_xp = Beta('B_ruta_xp', 0, None, None, 0)
B_bike_xp = Beta('B_bike_xp', 0, None, None, 0)
B_bike_freq = Beta('B_bike_freq', 0, None, None, 0)
B_estudia = Beta('B_estudia', 0, None, None, 0)
B_trabaja = Beta('B_trabaja', 0, None, None, 0)

# Thresholds for SCL metrics
t_SCL_mean = Beta('t_SCL_mean', 1, None, None, 0)
t_SCR_freq = Beta('t_SCR_freq', 1, None, None, 0)
t_SCL_AMP = Beta('t_SCL_AMP', 1, None, None, 0)
t_SCL_RT = Beta('t_SCL_RT', 1, None, None, 0)
t_SCL_MAX = Beta('t_SCL_MAX', 1, None, None, 0)
t_speed = Beta('t_speed', 1, None, None, 0)

# Gains for SCL metrics
g_SCL_mean = Beta('g_SCL_mean', 1, None, None, 0)
g_SCR_freq = Beta('g_SCR_freq', 1, None, None, 0)
g_SCL_AMP = Beta('g_SCL_AMP', 1, None, None, 0)
g_SCL_RT = Beta('g_SCL_RT', 1, None, None, 0)
g_SCL_MAX = Beta('g_SCL_MAX', 1, None, None, 0)
g_speed = Beta('g_speed', 1, None, None, 0)

# Assume that we also need to define variability (sigma) for each SCL-related measure
sigma_SCL_mean = Beta('sigma_SCL_mean', 1, 1.0e-5, None, 0)
sigma_SCR_freq = Beta('sigma_SCR_freq', 1, 1.0e-5, None, 0)
sigma_SCL_AMP = Beta('sigma_SCL_AMP', 1, 1.0e-5, None, 0)
sigma_SCL_RT = Beta('sigma_SCL_RT', 1, 1.0e-5, None, 0)
sigma_SCL_MAX = Beta('sigma_SCL_MAX', 1, 1.0e-5, None, 0)
sigma_speed = Beta('sigma_speed', 1, 1.0e-5, None, 0)

# Define the utility function based on specified variables and betas
LAT_PPSE = (Const +
            CO2 * B_CO2 +
            bright * B_bright +
            noise * B_noise +
            temp * B_temp +
            hum * B_hum +
            temphum * B_temphum +
            
            distance_to_int * B_distance_to_int +
            pendiente * B_pendiente +
            time * B_time +
            time_of_day * B_time_of_day +

            inter_paso_cebra * B_inter_paso_cebra + 
            inter_ceda_paso* B_inter_ceda_paso + 
            inter_pare * B_inter_pare + 
            
            s_ciclovia * B_s_ciclovia +
            park * B_park +
            ancho * B_ancho +
            bid_si * B_bid_si +
            car_way_si * B_car_way_si +
            park_si * B_park_si +
            nlanes * B_nlanes +
            
            gen * B_gen +
            edad * B_edad +
            ruta_xp * B_ruta_xp +
            bike_xp * B_bike_xp +
            bike_freq * B_bike_freq +
            trabaja * B_trabaja +
            estudia * B_estudia +
            # Here 'eta' needs to be defined somewhere in your model as well.
            bioDraws('eta', 'NORMAL_MLHS'))


#Measurenent equations

# Measurement equations incorporating the stochastic draws and utility function
ySCL_mean = (g_SCL_mean + t_SCL_mean * bioDraws('eta_SCL_mean', 'NORMAL_MLHS')) * LAT_PPSE
ySCR_freq = (g_SCR_freq + t_SCR_freq * bioDraws('eta_SCR_freq', 'NORMAL_MLHS')) * LAT_PPSE
ySCL_AMP = (g_SCL_AMP + t_SCL_AMP * bioDraws('eta_SCL_AMP', 'NORMAL_MLHS')) * LAT_PPSE
ySCL_RT = (g_SCL_RT + t_SCL_RT * bioDraws('eta_SCL_RT', 'NORMAL_MLHS')) * LAT_PPSE
yspeed = (g_speed + t_speed * bioDraws('eta_speed', 'NORMAL_MLHS')) * LAT_PPSE
ySCL_MAX = (g_SCL_MAX + t_SCL_MAX * bioDraws('eta_SCL_MAX', 'NORMAL_MLHS')) * LAT_PPSE

# Probability of observing each physiological measure given the model predictions and variability
P_SCL_mean = bioNormalCdf((SCL_mean - ySCL_mean) / sigma_SCL_mean)
P_SCR_freq = bioNormalCdf((SCR_freq - ySCR_freq) / sigma_SCR_freq)
P_SCL_AMP = bioNormalCdf((SCL_AMP - ySCL_AMP) / sigma_SCL_AMP)
P_SCL_RT = bioNormalCdf((SCL_RT - ySCL_RT) / sigma_SCL_RT)
P_speed = bioNormalCdf((speed - yspeed) / sigma_speed)
P_SCL_MAX = bioNormalCdf((SCL_MAX - ySCL_MAX) / sigma_SCL_MAX)


# Conditional to eta, we have a logit model (called the kernel) for

# Conditional to etas, we have the product of ordered probit for the
# indicators.
condlike = (P_SCL_mean
    * P_SCR_freq
    * P_SCL_AMP
    * P_SCL_RT
    * P_speed
    * P_SCL_MAX
)

# We integrate over omega using numerical integration
loglike = log(MonteCarlo(condlike))

# These notes will be included as such in the report file.
# userNotes = (
#     'Latent variable model explaining latent emotion (instant utility?), '
#     'with environmental variables and transportation attributes'
#     'while measuring it with psychophisiological indicators and stated emotions'
# )

# Define level of verbosity
logger = logging.getLogger('biogeme')
logging.basicConfig(level=logging.INFO)  # Or any other logging level you need


# Create the Biogeme object
biogeme = bio.BIOGEME(database, loglike, numberOfDraws=100)
biogeme.modelName = '05latentChoiceFull_mc_v5'

biogeme.estimate()