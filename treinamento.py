import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def preparar_dados(file_path):
    base = pd.read_csv(file_path)

    base['Size'] = base['Size'].astype(str).str.replace(',', '.').astype(float)

    X = base[['Weight', 'Size']].values
    y = base['Class'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, 'label_encoder.pkl')

    return X, y

def criar_modelo():
    model = Sequential([
        Dense(8, input_shape=(2,), activation='relu'),
        Dense(6, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def treinar_modelo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = criar_modelo()

    checkpoint = ModelCheckpoint('model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[checkpoint])
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')
    return accuracy

def carregar_modelo():
    model = tf.keras.models.load_model('model.keras')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, scaler, label_encoder

def prever_especie(input_data):
    model, scaler, label_encoder = carregar_modelo()
    input_data = scaler.transform([input_data])
    y_pred = model.predict(input_data)
    y_pred_class = np.argmax(y_pred, axis=1)
    species = label_encoder.inverse_transform(y_pred_class)
    return species[0]

if __name__ == '__main__':
    file_path = 'apples_and_oranges.csv' 
    X, y = preparar_dados(file_path)
    treinar_modelo(X, y)
    sample_input = [72, 5.85] 
    species = prever_especie(sample_input)
    print(f'A fruta prevista para {sample_input} é: {species}')
