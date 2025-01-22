import logging
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Log para verificar rutas registradas
logging.info(f"Rutas registradas: {app.url_map}")

# Cargar el modelo previamente entrenado
modelo = joblib.load('modelo_regresion_lineal.joblib')

@app.route('/')
def home():
    return "API del predictor de sueño funcionando correctamente"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        valores = np.array([data['WorkoutTime'], data['ReadingTime'], data['PhoneTime'],data['SleepTime'],
                            data['WorkHours'], data['CaffeineIntake'], data['RelaxationTime']]).reshape(1, -1)
        prediccion = modelo.predict(valores)[0]

        if prediccion < 6:
            estado = "Mala rutina de sueño"
            recomendaciones = [
                "Reduce el consumo de cafeína después de las 5 PM.",
                "Evita el uso de dispositivos electrónicos antes de dormir.",
                "Establece un horario de sueño regular."
            ]
        else:
            estado = "Buena rutina de sueño"
            recomendaciones = [
                "Continúa con tus buenos hábitos.",
                "Intenta mantener tu nivel de actividad física."
            ]

        return jsonify({"estado": estado, "prediccion": prediccion, "recomendaciones": recomendaciones})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
