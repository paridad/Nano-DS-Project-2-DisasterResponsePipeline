# Nano-DS-Project-2-DisasterResponsePipeline
Holds Project Data Files and ETL, Model scripts

## 1. Installation 


The following additional packages need to be installed. The code uses Python 3.7

		 - punkt
		 - wordnet
	 	 - stopwords
 
		Example:
		*	nltk.download('wordnet')
		*	nltk.download('punkt')
		*	nltk.download('stopwords')

## 2.	Project Motivation

In this project we have analyzed thousands of real messages provided by Figure 8, sent during natural disasters either via social media or directly to disaster response organizations.  I have built an ETL pipeline that processes message and category data from csv files and load them into a SQLite database. Then the  machine learning pipeline reads the data from SQLite database, create and save a multi-output supervised learning model. Finally the  web app will extract data from this database to provide data visualizations and use the model to classify new messages for 36 categories. 
Machine learning is critical to helping different organizations understand which messages are relevant to them and which messages to prioritize.  In this project, I have used the skills related to ETL pipelines, natural language processing, and machine learning pipelines to create the Model to classify the messages that would have  real world impact.


## 3.	File Descriptions

    The project  work is  housed across  three  folders. All the data /scripts/models can be accessed in GitHub using following URL;

	    GitHub URL: https://github.com/paridad/Nano-DS-Project-2-DisasterResponsePipeline

        data 
    	   - disaster_categories.csv: categories dataset 
     	   - disaster_messages.csv:  messages dataset 
     	   - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
               - DisasterResponse.db:ETL pipeline output , i.e. SQLite DB containing cleaned combined data of messages and categories
			
       models
          - train_classifier.py: Machine learning pipeline script to  create, improve,train  and export the model
          - classifier.pkl: output of the machine learning pipeline, i.e. a trained classifier
			 
       app
    	  - run.py: Flask file to run the web application
    	  - templates: This contains html files needed to render the results for Web Application



## 4.	Instructions:
 
     Run the following commands in the project's root directory to set up your database and model.
	
	 1. To run ETL pipeline that cleans data and stores in database: python data/process_data.py data/disaster_messages.csv  
		    data/disaster_categories.csv data/DisasterResponse.db
	
     2. To run ML pipeline that trains classifier and saves python :python models/train_classifier.py data/DisasterResponse.db  
         models/classifier.pkl

     3. 	Run the following command in the app's directory to run your web app. python run.py
    
	 4.   Go to http://localhost:3001/


## 5.	RESULTS 

I have  selected  a Supervised ML  “Random Forest Classification” model  to classify the disaster messages into  one or more of 36 categories.


**ETL Pipeline output**- Creates the SQLite DisasterReponse.DB

**Machine learning Pipeline** – Creates the classifier.pkl, which will be used by Web app for message classification

**Flask app**- This loads the data and model and  start the web app @  http://0.0.0.0:3001/ to accepts user query/message for classification. Below is the sample of message classification


![Visuaization-1](https://github.com/paridad/Nano-DS-Project-2-DisasterResponsePipeline/blob/master/Visualization-1.svg)

![Visualization-2](https://github.com/paridad/Nano-DS-Project-2-DisasterResponsePipeline/blob/master/Visualization-2.svg)

![Visualization-3](https://github.com/paridad/Nano-DS-Project-2-DisasterResponsePipeline/blob/master/Visualization-3.png)

![Visualization-4](https://github.com/paridad/Nano-DS-Project-2-DisasterResponsePipeline/blob/master/Visualization-4.jpg)



## 6.	Licensing, Authors, Acknowledgements 
I would like to  give credit to Udacity online courses   and to Figure-Eight.com for providing the data. 
Few additional  links/resources  are stated below that I have used as reference to complete my project.

 -	Choosing the right metric  Choosing the Right Metric for Evaluating Machine Learning Models-https://www.kdnuggets.com/2018/04/right-metric-evaluating-machine-learning-models-1.html
 
 -	Article  about What metrics should be used for evaluating a model on an imbalanced data set-https://towardsdatascience.com/what-metrics-should-we-use-on-imbalanced-data-set-precision-recall-roc-e2e79252aeba
 
 -	Article on various techniques of the data exploration process-https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
 
 -	Data Mining for Business Analytics- Galit Shmueli, pertr C Bruce ,Wiley Publications

