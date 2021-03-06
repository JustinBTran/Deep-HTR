import os
import shutil

from flask import Flask, redirect, render_template, request
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')

from pic_inference import *

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

app = Flask(__name__, static_url_path="/static")
Bootstrap(app)
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
# limit upload size upto 8mb
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/app/', methods=['GET', 'POST'])
def main():
    print(request.form)
    cont = None
    postal_codes = []
    if request.method == 'POST':
        stat_info = request.form.to_dict()

        if 'doc' not in request.files:
            print('No file attached in request')
            return redirect(request.url)

        doc_object = request.files['doc']

        doc_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(doc_object.filename))

        try:
            shutil.rmtree(app.config['UPLOAD_FOLDER'])
            os.mkdir(app.config['UPLOAD_FOLDER'])
        except:
            os.mkdir(app.config['UPLOAD_FOLDER'])

        doc_object.save(doc_path)

        segment(doc_path)

        words = 'lines'
        paths = []

        model_path = './models/trocr-handwritten-best.pt'

        beam = 5
        model, cfg, task, generator, bpe, img_transform, device = init(model_path, beam)

        cont = []
        for image in sorted(os.listdir(words)):
            if image == '.DS_Store':
                continue
            f = os.path.join(words, image)
            paths.append(f)
        paths.sort(key=os.path.getctime)

        for path in paths:
            cont.append(read_word_trocr(path, model, cfg, generator, bpe, img_transform)[1:])

    return render_template('index.html', cont=' '.join(cont) if cont else None)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port)
