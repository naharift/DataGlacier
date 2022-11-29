from flask import Flask,render_template,url_for,request
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df1 = pd.read_csv("dataset/Youtube02-KatyPerry.csv")        
    df2 = pd.read_csv("dataset/Youtube04-Eminem.csv")           
    df3 = pd.read_csv("dataset/Youtube05-Shakira.csv")

    # Merge all the datasset into single file
    frames = [df1,df2, df3]                          
    df_merged = pd.concat(frames)
    keys = ["Katy","Eminem", "Shakira"]
    df_with_keys = pd.concat(frames,keys=keys)
    dataset=df_with_keys

    # working with text content
    dataset = dataset[["CONTENT" , "CLASS"]]             # context = comments of viewers & Class = ham or Spam

    # Predictor and Target attribute
    dataset_X = dataset['CONTENT']                       # predictor attribute
    dataset_y = dataset['CLASS']                         # target attribute

    # Extract Feature With TF-IDF model 
    corpus = dataset_X                               # declare the variable
    cv = TfidfVectorizer()                           # initialize the TF-IDF  model
    X = cv.fit_transform(corpus).toarray()           # fit the corpus data into BOW model


    # import pickle file of my model
    model = open("model/model.pkl","rb")
    clf = pickle.load(model)
    
    if request.method == 'POST':
    	comment = request.form['comment']
    	data = [comment]
    	vect = cv.transform(data).toarray()
    	my_prediction = clf.predict(vect)
    	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)