import requests
url = 'http://localhost:5000/results'
r = requests.post(url,json={'Temperatura Media(C)':24,
 'Temperatura Minima(C)':21.1, 
 'Temperatura Maxima(C)':28.2,
 'Precipitacao(mm)':13.6,
 'Final de Semana':1.0})
print(r.json())
