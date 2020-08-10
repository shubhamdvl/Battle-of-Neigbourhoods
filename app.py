from flask import Flask
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV 
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from flask import render_template
from flask import request
import pickle


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


# Function to find the kth best neighbourhood

# def find_best(ans,rk):
#   ans_index = delhi_final_groupped[delhi_final_groupped['Neighborhood'] == ans]['index']
#   ans_index = int(ans_index)
#   sim_sorted = np.sort(sim[ans_index][0])[::-1]
#   sim_list = list(sim[ans_index][0])
#   mosts = sim_list.index(sim_sorted[int(rk)-1])
#   nei = delhi_final_groupped.iloc[mosts]['Neighborhood']
#   price = delhi_final_groupped.iloc[mosts]['Prices']
#   return nei, price

  
@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())[0]
        ans = to_predict_list
        prices = pd.read_csv("prices_final.csv")
        delhi_final_groupped = pd.read_csv("delhi_cos.csv")
        delhi_final_groupped = pd.merge(delhi_final_groupped, prices, on='Neighborhood')

        sim = cosine_similarity(delhi_final_groupped.drop(['Neighborhood','Prices','Unnamed: 0_x'],1))
        sim[sim >= 1] = -2
        delhi_final_groupped = delhi_final_groupped.reset_index()

        ans_index = delhi_final_groupped[delhi_final_groupped['Neighborhood'] == ans]['index']
        ans_index = int(ans_index)
        sim_sorted = np.sort(sim[ans_index])[::-1]
        sim_list = list(sim[ans_index])
        mosts = sim_list.index(sim_sorted[1])
        mosts2 = sim_list.index(sim_sorted[2])
        mosts3 = sim_list.index(sim_sorted[3])
        neig1 = delhi_final_groupped.iloc[mosts]['Neighborhood']
        pric1 = delhi_final_groupped.iloc[mosts]['Prices']
        neig2 = delhi_final_groupped.iloc[mosts2]['Neighborhood']
        pric2 = delhi_final_groupped.iloc[mosts2]['Prices']
        neig3 = delhi_final_groupped.iloc[mosts3]['Neighborhood']
        pric3 = delhi_final_groupped.iloc[mosts3]['Prices']


        
        return render_template("index.html", neigh1 = neig1, price1 = pric1, neigh2 = neig2, price2 = pric2,neigh3 = neig3, price3 = pric3)
if __name__ == "__main__":
    app.run(debug=True)