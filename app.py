from flask import Flask, render_template, request
from treinamento import preparar_dados, treinar_modelo, prever_especie
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            X, y = preparar_dados(file_path)
            accuracy = treinar_modelo(X, y)
            return render_template('index.html', accuracy=round(accuracy * 100, 2))
        else:
            weight = float(request.form['weight'])
            size = float(str(request.form['size']).replace(',', '.'))
            input_data = [weight, size]
            species = prever_especie(input_data)
            return render_template('index.html', species=species)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
