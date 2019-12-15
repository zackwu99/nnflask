from flask import Flask, render_template, request, jsonify
from lib import parameters
from lib import neural_network as nn


app = Flask(__name__)


network: nn.NeuralNetwork = nn.NeuralNetwork(parameters.input_size, parameters.num_hidden_layer,
                                             parameters.hidden_node_size, parameters.output_size,
                                             parameters.learning_rate)

network.load_weights(parameters.input_layer_weight_file, parameters.output_layer_weight_file,
                     parameters.hidden_layer_weight_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/guess_number', methods=['POST'])
def get_guess_number():
    img_str = request.form["image_str"]
    (percent, num) = network.check_one_row(img_str);
    the_number: int = num
    the_percent: float = percent[0]
    round_percent = round(the_percent*10000)/100;
    resp = {'answer': the_number, "percent": round_percent}
    return jsonify(resp)


app.run(port=9305)


if __name__ == '__main__':
    app.run(debug=True)


