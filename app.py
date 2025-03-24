from flask import Flask, jsonify,request, render_template
import clustering
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route("/", methods=["POST"])
def receive_data():
    received_data = request.get_data()
    received_data = received_data.decode("utf-8").split("\"")
    file_path = "https://drive.google.com/file/d/1MVbMePjzEsqezuFus6MVAIFY7739WufQ/view?usp=sharing" if received_data[3] =="GSE108474" else ""
    clusters,data0, data1 = clustering.main(file_path) 
    data0 = ",".join([str(data) for data in data0])
    data1 = ",".join([str(data) for data in data1])
    return jsonify(data0),jsonify(data1),jsonify(clusters)

if __name__ == '__main__':
    app.run(debug=True)
