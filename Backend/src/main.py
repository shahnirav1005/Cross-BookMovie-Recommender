from flask import Flask, request, json, jsonify, abort
import mysql.connector
from flask_cors import CORS, cross_origin
import recommend


rec_data = recommend.setup()

app = Flask(__name__)

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response

with app.app_context(): 
    if __name__ == '__main__':
        app.run(debug=True)

    db = mysql.connector.connect(user='dvauser', host='localhost', database='recommender')


    def assert_json(code):
        if not request.is_json: # seems to only check header for 'application/json'
            abort(code)
        try:
            request.get_json() # throws a 400 if json payload is invalid
        except:
            abort(code)


    @app.route('/user', methods=['POST'])
    def create_user():
        """Add a new user"""
        cursor = db.cursor()
        assert_json(405)
        cursor.execute("USE recommender;")
        print("Trying");   
        print(request.get_json());

        moviesHistory = request.get_json().get('moviesHistory')
        booksHistory = request.get_json().get('booksHistory')

        list_movies = moviesHistory.split("||")
        print(list_movies[0])
        list_books = booksHistory.split("||")
        genre_books_history = recommend.get_grenre_books(list_books,rec_data["book_info"])
        genre_movies_history = recommend.get_grenre_movies(list_movies,rec_data["movie_data_final"])

        insert_statement = (
            "INSERT INTO User (username, password, booksHistory, moviesHistory, booksGenresHistory, moviesGenresHistory) VALUES (%s, %s, %s, %s, %s, %s)"
        )

        data=(request.get_json().get('username'),request.get_json().get('password'),request.get_json().get('booksHistory'),request.get_json().get('moviesHistory'), genre_books_history, genre_movies_history)
        cursor.execute(insert_statement, data)
        db.commit()
        return ""


    @app.route('/user/addBookHistory/<username>/<book_id>', methods=['GET'])
    def add_book_user(username,book_id):
        """Add a new user"""
        book_info = rec_data["book_info"]
        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("SELECT booksHistory FROM User WHERE username=%s;",(username,))
        data=cursor.fetchone()
        cursor.close()

        list_books = data['booksHistory'].split("||")
        list_books.pop(0)
        book = book_info.loc[book_info['book_id'] == int(book_id)].title

        list_books.append(book.values[0])
        bookHistory = list_books[0]+"||"+list_books[1]+"||"+list_books[2]+"||"+list_books[3]+"||"+list_books[4]

        genre_books_history = recommend.get_grenre_books(list_books,book_info)

        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("UPDATE User SET booksHistory = %s WHERE username=%s;",(bookHistory,username,))
        cursor.execute("UPDATE User SET booksGenresHistory = %s WHERE username=%s;",(genre_books_history,username,))
        db.commit()
        cursor.close()
        return ""

    @app.route('/user/addMovieHistory/<username>/<movie_id>', methods=['GET'])
    def add_movie_user(username,movie_id):
        """Add a new user"""
        movie_data_final = rec_data["movie_data_final"]
        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("SELECT moviesHistory FROM User WHERE username=%s;",(username,))
        data=cursor.fetchone()
        cursor.close()

        list_movies = data['moviesHistory'].split("||")
        list_movies.pop(0)
        movie = movie_data_final.loc[movie_data_final['movie_id'] == int(movie_id)].title

        list_movies.append(movie.values[0])
        movieHistory = list_movies[0]+"||"+list_movies[1]+"||"+list_movies[2]+"||"+list_movies[3]+"||"+list_movies[4]

        genre_movies_history = recommend.get_grenre_movies(list_movies,movie_data_final)

        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("UPDATE User SET moviesHistory = %s WHERE username=%s;",(movieHistory,username,))
        cursor.execute("UPDATE User SET moviesGenresHistory = %s WHERE username=%s;",(genre_movies_history,username,))
        db.commit()
        cursor.close()
        return ""


    @app.route('/user/<username>', methods=['GET'])
    def get_user(username):
        """Get a user"""
        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("SELECT username, password, booksHistory, moviesHistory FROM User WHERE username=%s;",(username,))
        data=cursor.fetchone()
        cursor.close()
        if data:
            return json.dumps(data)
        else:
            abort(404)

    @app.route('/books', methods=['GET'])
    def get_books():
        """Get list of all books"""
        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("SELECT * FROM Books LIMIT 2000")
        data=cursor.fetchall()
        cursor.close()
        if data:
            return json.dumps(data)
        else:
            abort(404)

    @app.route('/movies', methods=['GET'])
    def get_movies():
        """Get list of all books"""
        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("SELECT * FROM Movies LIMIT 2000")
        data=cursor.fetchall()
        cursor.close()
        if data:
            return json.dumps(data)
        else:
            abort(404)


    @app.route('/user/bookrecomm/<username>/<book_id>', methods=['GET'])
    def get_book_recomm_user(username,book_id):
        """Get book recommendation for specified user"""
        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("SELECT moviesGenresHistory FROM User where username=%s;",(username,))
        data=cursor.fetchall()
        cursor.close()

        out = recommend.get_book_recommendation_user(data[0]['moviesGenresHistory'],int(book_id),rec_data["book_info"],rec_data["count_vect_movies"],rec_data["X_train_tfidf_movies"],rec_data["movie_data_final"])
        if out:
            return json.dumps(out)
        else:
            abort(404)



    @app.route('/user/bookrecomm/<book_id>', methods=['GET'])
    def get_book_recomm(book_id):
        """Get book recommendation for specified book"""
        out = recommend.get_book_recommendation(int(book_id),rec_data["book_info"],rec_data["count_vect_movies"],rec_data["X_train_tfidf_movies"],rec_data["movie_data_final"])
        if out:
            return json.dumps(out)
        else:
            abort(404)

    @app.route('/user/movierecomm/<username>/<movie_id>', methods=['GET'])
    def get_movie_recomm_user(username,movie_id):
        """Get movie recommendation for speicifed user"""
        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("SELECT booksGenresHistory FROM User where username=%s;",(username,))
        data=cursor.fetchall()
        cursor.close()

        out=recommend.get_movie_recommendation_user(data[0]['booksGenresHistory'],int(movie_id),rec_data["movie_data_final"],rec_data["count_vect_books"],rec_data["X_train_tfidf_books"],rec_data["tfidf_transformer_books"],rec_data["clf"],rec_data["book_info"])

        if out:
            return json.dumps(out)
        else:
            abort(404)

    @app.route('/user/movierecomm/<movie_id>', methods=['GET'])
    def get_movie_recomm(movie_id):
        """Get movie recommendation for speicifed movie"""
        out=recommend.get_movie_recommendation(int(movie_id),rec_data["movie_data_final"],rec_data["count_vect_books"],rec_data["X_train_tfidf_books"],rec_data["tfidf_transformer_books"],rec_data["clf"],rec_data["book_info"])
        if out:
            return json.dumps(out)
        else:
            abort(404)


    @app.route('/books/<bookid>', methods=['GET'])
    def get_book_details(bookid):
        """Get details of the specified book"""
        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("SELECT * FROM Books where book_id=%s;",(float(bookid),))
        data=cursor.fetchall()
        cursor.close()
        if data:
            return json.dumps(data)
        else:
            abort(404)

    @app.route('/movies/<movieid>', methods=['GET'])
    def get_movie_details(movieid):
        cursor = db.cursor(dictionary=True)
        cursor.execute("USE recommender;")
        cursor.execute("SELECT * FROM Movies where movie_id=%s;",(float(movieid),))
        data=cursor.fetchall()
        cursor.close()
        if data:
            return json.dumps(data)
        else:
            abort(404)




















            