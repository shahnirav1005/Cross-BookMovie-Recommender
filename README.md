# Cross-BookMovie-Recommender

## Important

Minor updates have been made this code which are not reflected in the README. This may affect the way the code needs to be run.
Make sure the database and the web portal are running behind the scenes before one starts trying to use the App. 
Quick setup is not bug-free and hence minor changes might be required based on the versions of installed requirements.

## Description

Page & Reel (Cross BookMovie Recommender) is an innovative app that provides users with a reliable method of cross media recommendation between books and movies. Users are required to craft a personalized account, containing their top 5 favorite movies and books. The user and their specified items will then be stored within our SQL database. Upon registration, users are provided a brief look at their homepage, containing all 10 selected items as well as several interactive tools. A seach bar is displayed at the top of the page to search the database for a given book or movie. Each item in the database is able to be run through our recommendation process which uses both the Naive Bayes "Bag-of-words" and Cosine Similarity Measure for the analysis of media text. In addition, we also filter recommended items based on the prefered genres of a given user.

Each item displayed to the user corresponds to a single recommendation. Users may filter the results by selection, for a drop down menu, the media type. Each book or movies is also linked, via a button, to Goodreads and IMDB, respectively. The user may choose to opt for recommending media contained in our database by clicking "Search via Page and Reel". Additional information regarding a given item is also displayed in each cell for the user's convenience, sich as description, genre, title, and writer(s). 

## Dependencies

- Python: 2.7
- MYSQL (Prefer Ver. 14 or higher)
- flask-mysqldb:0.2.0
- Flask:0.12.2
- Flask-SQLAlchemy:2.3.2
- mysql-connector:2.1.6x

## Quick Setup

To run the web service open a terminal at Backend folder and run the following commands:
1. pip install -r requirements.txt
2. cd src/
3. mysql < createDatabase.sql
4. FLASK_APP=main.py flask run --port=8888
5. Within main.py, please change line 22 to:
	db = mysql.connector.connect(user='[your mysql username]', password='[your sql password]', host='localhost', database='recommender')

## Instructions

To run the web portal, you will have to run a webserver.
Open a terminal at the WebPortal folder and run:
python -m SimpleHTTPServer 8000

The webserver can now be access at http://localhost:8000/login.html

## Usage
1. If you currently have an account in our database, enter your Username and Password in login.html. If not, please traverse to the registration page by clicking "Create and account->".
2. Please enter a username and password for the first step of registration and hit "Next" to continue.
3. From our drop-down menu, please select 5 of your favorite books and movies containined within our database. Hit "Register" to continue to your personalized homepage. 
4. You have several options here. For each item, you may select Go To Goodreads or Go to IMDB to view the item on their respective webpage. In addition, you may click Search Via Page and Reel to run the recommendation process for the selected item. This option will provide you with the recommended items for the opposite media type.
5. At the top of the screen, you may select either books or movies as a search parameter (please refresh the page upon re-querying our database with this method). Then, select from our search list the item of choice and hit go. This will provide the recommendations for the opposite media type of what you selected. 
6. Perform basic filtering of the results by selecting, at the upper left part of the page, Both, Books, or Movies. 
