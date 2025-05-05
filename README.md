# 🍎 Neural Training - Frutas (Maçãs e Laranjas)

Bem-vindo ao repositório **Neural Training Frutas**, este projeto utiliza redes neurais para classificar frutas (especificamente **maçãs** e **laranjas**) com base em características físicas como **peso** e **tamanho**.

# Integrantes
- [**Claudia Santana**](https://github.com/claudiasanttana)
- [**Mariana Siqueira**](https://github.com/Marigsiqueira)
- [**Thiago Messias**](https://github.com/Thmsantos)

## 📊 Visão Geral

O objetivo deste projeto é construir e treinar um modelo de aprendizado de máquina para prever o tipo da fruta com base em dados reais. O modelo aprende com dados históricos e pode prever corretamente novas amostras.

### Características do Projeto:
- **Treinamento de Modelo**: Rede neural construída com TensorFlow e Keras.
- **Padronização de Dados**: Uso de `StandardScaler` para normalizar as entradas.
- **Codificação de Classes**: Conversão de rótulos de frutas em valores numéricos com `LabelEncoder`.
- **Persistência de Modelos**: Os arquivos do modelo, scaler e encoder são salvos (`model.keras`, `scaler.pkl`, `label_encoder.pkl`) para reuso.
- **Interatividade**: Permite a previsão da fruta com base em entradas manuais como `[peso, tamanho]`.

## 🛠️ Tecnologias Necessárias
- Python - 3.10

## 🛠️ Tecnologias Utilizadas

- **Python**: Linguagem principal do projeto.
- **Pandas**: Para leitura e manipulação dos dados do CSV.
- **NumPy**: Para operações numéricas.
- **Scikit-learn**: Para pré-processamento (escalonamento e codificação).
- **TensorFlow / Keras**: Para construção, treinamento e avaliação da rede neural.
- **Joblib**: Para salvar e carregar objetos como o scaler e o encoder.
- **Flask (opcional)**: Pode ser usado para criar uma API ou interface web para o modelo.

## 🚀 Como Executar

1. **Clone o repositório:**
```bash
git clone https://github.com/Thmsantos/apples_and_oranges_classification
cd apples_and_oranges_classification
```

2. **Crie e ative um ambiente virtual com Python 3.10:**
```
py -3.10 -m venv venv
.\venv\Scripts\activate
```

3. **Instale as dependências**
```
pip install -r requirements.txt
```

4. **Execute o script**
```
python app.py
```

## 🚀 Como Acessar

[http://localhost:5000](http://localhost:5000) ou [http://127.0.0.1:5000](http://127.0.0.1:5000)


