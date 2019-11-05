import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Heatmap,Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)
print("data  is loaded")

# load model
model = joblib.load("../models/classifier.pkl")
print("model is loaded")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # data for All Message Category Counts Bar graphs

    cat_sums = df.iloc[:, 4:].sum().sort_values()
    cat_names = list(cat_sums.index)

    # data for Top 10 Message Category Counts Bar graphs

    top_10_cat_sums = df.iloc[:, 4:].sum().sort_values(ascending=False)[1:11]
    top_10_cat_names = list(top_10_cat_sums.index)
    
    # Prepare Data for Heat map
    # Compute pairwise  correlation values of Catgegories
    cat_heatmap = df.iloc[:,4:].corr().values
    cat_heatmap_names = list(df.iloc[:,4:].columns)

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
     
        # Visual 2 -Add new horizontal bar graph Category counts distribution   
        {

            'data': [

                Bar(

                    x=cat_sums,

                    y=cat_names,
                    orientation = 'h'
                  
                    
                    
                )

            ],



            'layout': {

                'title': 'Distribution of All Message Categories',

                'yaxis': {

                    'title': "Category"

                },

                'xaxis': {

                    'title': "count"

                },

            }

        },
   # Visual 3- Distribution of Top 10  Message categories
        {

            'data': [

                Bar(

                    x=top_10_cat_names,
                    y=top_10_cat_sums
                    
                  
                    
                    
                )

            ],



            'layout': {

                'title': 'Distribution of Top 10 Message Categories',

                'yaxis': {

                    'title': "Count"

                },

                'xaxis': {

                    'title': "Categories"

                },

            }

        },
        
     # Visual 4 -Add  Correlation Heat map of Categories   
        {

            'data': [

                Heatmap(

                    x=cat_heatmap_names,
                    y=cat_heatmap_names[::-1],
                    z=cat_heatmap
                  
                    
                    
                )

            ],



            'layout': {

                'title': 'Heatmap- Correlation of  Message Categories',

                'yaxis': {

                    'title': "Category"

                },

                'xaxis': {

                    'tickangle': -45

                },

            }

        },
        
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    print(query)
    print(classification_labels)
    print(classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)
  #  app.run(host='127.0.0.1', port=8000, debug=True)


if __name__ == '__main__':
    main()