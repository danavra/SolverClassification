from flask import Flask, request, jsonify
import json

# Terminal Comnand: FLASK_APP=server.py flask run
app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test():
 print("Example of data received")
 rf=request.form
 print(rf)
 resp = Response("Data received")
 resp.headers['Access-Control-Allow-Origin']='*'
 return resp

@app.route('/solve_problem', methods=['POST'])
def solve_problem():
 print("Running function solve_problem()")
 # the data
 rf=request.form
 print('data recieved: ' + str(rf) )

#  data_dic=json.loads(data)


 ans={'response' : 'server says hello'}
 resp = jsonify(ans)
 resp.headers['Access-Control-Allow-Origin']='*'
 return resp