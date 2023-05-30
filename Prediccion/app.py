from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import SCORERS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

df_original =pd.read_csv('./MedalTable (1).csv',sep=';')

df_original.loc["1247"] = [0, 1, 2, 3, 2020, 0, 12, 'ARG', 2016, 2012, 'Total Geral', 3, 4, 1, 4, 0, 0, 1, 1, 1020]
df=df_original[(df_original["FlagY1"]==1) & (df_original["FlagY2"]==1)]
df=df.drop(["Silver", "Bronze", "Aux", "Country", "Year-1", "Year-2", "Year+1","FlagY1", "FlagY2"], axis=1)

df_validate=df_original[df_original["Year"]==2016][["Home", "TotalYear-1", "TotalYear-2", "Total"]]
df=df[df["Year"]<2016]
X=df[["Home", "TotalYear-1", "TotalYear-2"]]
y=df["Total"]

x_test = df_validate[["Home", "TotalYear-1", "TotalYear-2"]]
y_test = df_validate["Total"]

scaler_test = MinMaxScaler()
scaler_test.fit(x_test)
x_test_scaled=scaler_test.transform(x_test)

df_validate=df_original[df_original["Year"]==2016][["Home", "TotalYear-1", "TotalYear-2", "Total"]]
df=df[df["Year"]<2016]
X=df[["Home", "TotalYear-1", "TotalYear-2",]]
y=df["Total"]

cv = model_selection.KFold(n_splits=10, shuffle=True, random_state=8)

scaler = MinMaxScaler()
scaler.fit(X)
X_scaled=scaler.transform(X)

parameters = {'kernel': ["linear", "rbf", "poly"], 'C': [1, 5, 10, 20, 50, 100], 'gamma': ["scale", "auto"]}

grid_svr = GridSearchCV(estimator=SVR(), param_grid=parameters, cv=5)
grid_svr.fit(X_scaled, y)
print("Best Parameters:",grid_svr.best_params_)
best_svr=grid_svr.best_estimator_

scoring='neg_mean_squared_error'
Mean_square_error_svr=model_selection.cross_val_score(best_svr, X_scaled, y, cv=cv, scoring=scoring)
scoring='r2'
r2_svr=model_selection.cross_val_score(best_svr, X_scaled, y, cv=cv, scoring=scoring)
confianza = r2_svr.mean()

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pais = request.form['pais']

        # Realiza aquí las operaciones o lógica que necesites con los datos ingresados

        filtro = (df_original['Country'] == pais) 
        base_pais = df_original[filtro]
        if pais == 'FRA':
            home = 1
        else:
            home = 0
        year_1 = base_pais.iloc[[-1]]['Total'].values[0]
        year_2 = base_pais.iloc[[-1]]['TotalYear-1'].values[0]

        new_data = []
        new_data+=[home,year_1,year_2]

        new_data_array = np.array(new_data)
        new_data_scaled = scaler.transform([new_data_array])

        RSULTADOS = grid_svr.predict(new_data_scaled) 
        medallas = RSULTADOS[0]

        return redirect('./resultado.html', pais=pais, medallas=medallas, confianza=confianza))
    return render_template('./home.html')

@app.route('./resultado.html')
def resultado():
            pais = request.args.get('pais')

            # Realiza aquí las operaciones o lógica que necesites con los datos ingresados

            filtro = (df_original['Country'] == pais) 
            base_pais = df_original[filtro]
            if pais == 'FRA':
                home = 1
            else:
                home = 0
            year_1 = base_pais.iloc[[-1]]['Total'].values[0]
            year_2 = base_pais.iloc[[-1]]['TotalYear-1'].values[0] 

            new_data = []
            new_data+=[home,year_1,year_2]

            new_data_array = np.array(new_data)
            new_data_scaled = scaler.transform([new_data_array])

            RSULTADOS = grid_svr.predict(new_data_scaled) 
            medallas = RSULTADOS[0]

            return render_template('./resultado.html', pais=pais, medallas=medallas, confianza=confianza)

if __name__ == '__main__':
    app.run()
    app.debug = True
