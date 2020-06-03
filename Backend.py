import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from DataUtil import directories_validation, make_groups
from FeatureExtraction import feature_extraction
from MetaFeatureExtractor import meta_feature_extractor
from Clustering import clustering
from run_all_experiments import run_all_experiments

app = Flask(__name__)
CORS(app)
print('********   SEVER IS LISTENING...   ********')


# solve_problem
@app.route('/solve_problem', methods=['POST'])
def solve_problem():
 print("Running function solve_problem()")
 uploaded_file = request.files['file']
 df = pd.read_csv(uploaded_file)
 print(df['Confidence'])
 x = list(df['Confidence'].to_numpy())
 print(x)
 # Server response

 ans = {'response' : 'ok' }
 resp = jsonify(ans)
 resp.headers['Access-Control-Allow-Origin']='*'
 return resp



# create_db
@app.route('/create_db', methods=['POST'])
def create_db():
 print("Running function create_db()")
 directories_validation()
 make_groups()
 feature_extraction()
 meta_feature_extractor()
 clustering()


 # Server response
 ans={'response' : 'server created a DB'}
 resp = jsonify(ans)
 resp.headers['Access-Control-Allow-Origin']='*'
 return resp











# db_analasys
@app.route('/db_analasys', methods=['POST'])
def db_analasys():
 print("***********************************\n  Running function db_analasys()\n***********************************")

 # the data
 rf = request.get_data()
 print('data recieved: ' + str(rf))
 data = eval(rf)


 # server response
 ans={'response' : str(data)}
 resp = jsonify(ans)
 resp.headers['Access-Control-Allow-Origin']='*'
 print('*********************************************************************************************************')
 return resp



if __name__ == '__main__':
    app.run()
