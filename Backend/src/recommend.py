import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict, defaultdict
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import math

def setup():
    movie_input=pd.read_csv('Movie_details.tsv',delimiter='\t', encoding="ISO-8859-1")
    book_input=pd.read_csv('Book_details.tsv',delimiter='\t', encoding="ISO-8859-1")

    #filtering out rows with null/nan
    mov_filtered=movie_input[movie_input.title.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.plot_movie.notnull()]
    mov_filtered=mov_filtered[mov_filtered.genres.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.rating.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.languages.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.director.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.poster.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.year.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.votes.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.writer.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.awards.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.runtime.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.country.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.released.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.languages.notnull()]
    # mov_filtered=mov_filtered[mov_filtered.production.notnull()]


    book_filtered=book_input[book_input.Author_names.notnull()]

    #Taking out the required columns-movies
    movie_info_col=mov_filtered["movie_id"]
    movie_info_col2=mov_filtered["plot_movie"]
    movie_info_col1=mov_filtered["title"]
    movie_info_col3=mov_filtered["genres"]
    movie_info_col4=mov_filtered["languages"]
    movie_info_col5=mov_filtered["rating"]
    movie_info_col6=mov_filtered["director"]
    movie_info_col7=mov_filtered["poster"]
    movie_info_col8=mov_filtered["year"]
    movie_info_col9=mov_filtered["votes"]
    movie_info_col10=mov_filtered["writer"]
    movie_info_col11=mov_filtered["awards"]
    movie_info_col12=mov_filtered["runtime"]
    movie_info_col13=mov_filtered["country"]
    movie_info_col14=mov_filtered["released"]
    movie_info_col15=mov_filtered["production"]
    movie_info=pd.concat([movie_info_col,movie_info_col1,movie_info_col2,movie_info_col3,movie_info_col4,movie_info_col5,movie_info_col6,
                        movie_info_col7,movie_info_col8,movie_info_col9,movie_info_col10,movie_info_col11,movie_info_col12,movie_info_col13,movie_info_col14,
                        movie_info_col15],axis=1)
    movie_genre_list= movie_info["genres"].unique()

    #GenreName in movie_genre_list can be tuples with multiple genres for the same record
    movie_genre_list=pd.DataFrame(data=movie_genre_list,columns=['GenreName'])
    abc=np.array(range(1, movie_genre_list.shape[0]+1))
    movie_genre_list["Target"]=pd.DataFrame(data=range((len(movie_genre_list))))
    movie_genre_list["Target"]=pd.DataFrame(data=range((len(movie_genre_list))))

    #Split csv in GenreName column
    new_moviegenrelist=movie_info_col3.apply(lambda x: pd.Series(x.split(',')))
    unique_genres=new_moviegenrelist.iloc[:, 0].unique()
    unique_genres=pd.DataFrame(data=unique_genres, columns=['Genre_Name'])

    #IMPORTANT- UNIQUE GENRES IN MOVIES ALONG WITH THEIR CLASS NAMES
    unique_genres["Target"]=pd.DataFrame(data=range((len(unique_genres))))
    #unique_genres.to_csv('Movies_Unique_Genres.tsv', sep='\t', encoding="utf-8")

    #Assign CLass value to target column
    genrecol=new_moviegenrelist
    mapping=dict(zip(unique_genres.Genre_Name,unique_genres.Target))
    temp=new_moviegenrelist.iloc[:,0].apply(lambda x: mapping[x])
    targetclass_for_movie_data_final=temp.to_frame(name='ClassVal')
    movie_data_final=pd.DataFrame(data=pd.concat([movie_info_col,movie_info_col1,movie_info_col2,genrecol,movie_info_col4,movie_info_col5,movie_info_col6,movie_info_col7,movie_info_col8
                    ,movie_info_col9,movie_info_col10,movie_info_col11,movie_info_col12,movie_info_col13,movie_info_col14,movie_info_col15,targetclass_for_movie_data_final],axis=1))
    movie_data_final=movie_data_final.drop_duplicates(keep='first')
    movie_data_final=movie_data_final.reset_index(drop=True)
    # movie_data_final.to_csv('Movie_Final_Used_For_Pred.tsv', sep='\t', encoding="ISO-8859-1")



    #Clean out nan titles
    book_filtered=book_filtered[book_filtered.title.notnull()]
    book_filtered=book_filtered[book_filtered.book_id.notnull()]
    #Clean out titles with special characters

    #Taking out the required columns-books
    book_info_col=book_filtered["book_id"]
    book_info_col1=book_filtered["title"]
    book_info_col2=book_filtered["Genres"]



    #Split csv in GenreName column
    new_bookgenrelist=book_info_col2.apply(lambda x: pd.Series(x.split(',')))
    split_book_genres=new_bookgenrelist.iloc[:, 0].unique()
    split_book_genres=pd.DataFrame(data=split_book_genres, columns=['Genre_Name_Book_Final'])

    #Extract the unique genres
    Unique_genres_book=split_book_genres.iloc[:,0].unique()
    Unique_genres_book=pd.DataFrame(data=Unique_genres_book)
    Unique_genres_book["Target"]=pd.DataFrame(data=range((len(Unique_genres_book))))

    mapping_books=dict(zip(Unique_genres_book.iloc[:,0],Unique_genres_book.Target))
    mapping_books_inverse=dict(zip(Unique_genres_book.Target,Unique_genres_book.iloc[:,0]))
    temp_book=new_bookgenrelist.iloc[:,0].apply(lambda x: mapping_books[x])
    targetclass_for_book_data_final=temp_book.to_frame(name='ClassVal')

    book_info_col3=book_filtered["Subjects"]
    book_info_col4=book_filtered["Description"]
    book_info_col5=book_filtered["Rating"]
    book_info_col6=book_filtered["Author_names"]
    book_info_col7=book_filtered["Publisher"]
    book_info_col8=book_filtered["isbn_13"]
    book_info_col9=book_filtered["Publish_date"]
    book_info_col10=book_filtered["url"]

    book_info=pd.concat([book_info_col,book_info_col1,split_book_genres,book_info_col3,book_info_col4,book_info_col5,book_info_col6,book_info_col7,book_info_col8,book_info_col9,book_info_col10,targetclass_for_book_data_final],axis=1)
    book_info=book_info.drop_duplicates(keep='first')
    book_info=book_info.reset_index(drop=True)
    book_info=book_info[book_info.title.notnull()]
    # book_info.to_csv('Book_Final_Used_For_Pred.tsv', sep='\t', encoding="utf-8")

    ## TF-IDF Calculation for all entries- movie
    count_vect_movies = CountVectorizer()
    X_train_counts = count_vect_movies.fit_transform(movie_data_final.plot_movie)
    tfidf_transformer_movies = TfidfTransformer()
    X_train_tfidf_movies = tfidf_transformer_movies.fit_transform(X_train_counts)

    ## TF-IDF Calculation for all entries- books
    count_vect_books = CountVectorizer()
    X_train_counts = count_vect_books.fit_transform(book_info.Description.values.astype('U'))
    tfidf_transformer_books = TfidfTransformer()
    X_train_tfidf_books = tfidf_transformer_books.fit_transform(X_train_counts)

    #################TRAIN NAIVE BAYES FOR BOOKS######################
    clf = MultinomialNB().fit(X_train_tfidf_books,book_info.ClassVal)
    data={}
    data["book_info"]=book_info
    data["count_vect_movies"] = count_vect_movies
    data["X_train_tfidf_movies"] = X_train_tfidf_movies
    data["movie_data_final"] = movie_data_final
    data["count_vect_books"] = count_vect_books
    data["X_train_tfidf_books"] = X_train_tfidf_books
    data["tfidf_transformer_books"] = tfidf_transformer_books
    data["clf"] = clf

    return data


def get_book_recommendation(book_id,book_info,count_vect_movies,X_train_tfidf_movies,movie_data_final):
        ###################################################
        #input=movie_data_final.loc[10051,:].plot_movie
        input=book_info.loc[book_id-1,:].Description
        input_title=book_info.loc[book_id-1,:].title
        input=[input]
        X_new_counts = count_vect_movies.transform(input)

        ###############CHECK SIMILARITY#####################
        check_similarity_with_each_record=cosine_similarity(X_new_counts,X_train_tfidf_movies)
        flattened_sim_scores=check_similarity_with_each_record.flatten()
        index_loc_of_related_descriptions=np.argsort(flattened_sim_scores)[::-1][:10]
        Confidence_values=flattened_sim_scores[index_loc_of_related_descriptions]
        j=0
        #Recommendations
        list_recommendations=[]
        for i in index_loc_of_related_descriptions:
            print "###################Prediction",j+1,"#####################"
            print movie_data_final.loc[i,:].title
            #print movie_data_final.loc[i,:].plot_movie
            j=j+1
            movie={}
            movie["movie_id"]=movie_data_final.loc[i,:].movie_id
            movie["title"]=movie_data_final.loc[i,:].title
            movie["plot_movie"]=movie_data_final.loc[i,:].plot_movie
            movie["genres"]=str(movie_data_final.loc[i,:][3])+","+str(movie_data_final.loc[i,:][4])+","+str(movie_data_final.loc[i,:][5])
            movie["rating"]=movie_data_final.loc[i,:].rating
            movie["director"]=movie_data_final.loc[i,:].director
            movie["poster"]=movie_data_final.loc[i,:].poster
            movie["languages"]=movie_data_final.loc[i,:].languages
            movie["year"]=movie_data_final.loc[i,:].year
            movie["votes"]=movie_data_final.loc[i,:].votes
            movie["writer"]=movie_data_final.loc[i,:].writer
            movie["awards"]=str(movie_data_final.loc[i,:].awards)
            movie["runtime"]=movie_data_final.loc[i,:].runtime
            movie["country"]=movie_data_final.loc[i,:].country
            movie["released"]=movie_data_final.loc[i,:].released
            movie["languages"]=movie_data_final.loc[i,:].languages
            movie["production"]=movie_data_final.loc[i,:].production

            # movie["plot_movie"]=movie_data_final.loc[i,:].plot_movie
            list_recommendations.append(movie)
        print "##################Completed###########################"
        return list_recommendations

def get_book_recommendation_user(user_genres,book_id,book_info,count_vect_movies,X_train_tfidf_movies,movie_data_final):
        ###################################################
        #input=movie_data_final.loc[10051,:].plot_movie
        input=book_info.loc[book_id-1,:].Description
        input_title=book_info.loc[book_id-1,:].title
        input=[input]
        X_new_counts = count_vect_movies.transform(input)

        ###############CHECK SIMILARITY#####################
        check_similarity_with_each_record=cosine_similarity(X_new_counts,X_train_tfidf_movies)
        flattened_sim_scores=check_similarity_with_each_record.flatten()
        index_loc_of_related_descriptions=np.argsort(flattened_sim_scores)[::-1][:10]
        Confidence_values=flattened_sim_scores[index_loc_of_related_descriptions]
        j=0

        #get genres of user
        list_genres = user_genres.split(",")
        while "nan" in list_genres: list_genres.remove('nan') 

        recommen_list = []
        count = 0
        z = 0;
        for i in index_loc_of_related_descriptions:
            if str(movie_data_final.loc[i,:][3]) in list_genres or str(movie_data_final.loc[i,:][4]) in list_genres or str(movie_data_final.loc[i,:][5]) in list_genres:
                if count < 5:
                    recommen_list.append(index_loc_of_related_descriptions[z])
                    count = count + 1
                    z = z + 1
                else:
                    break
            else:
                z = z + 1

        for i in range(0,10):
            if count >=5:
                break
            else:
                if index_loc_of_related_descriptions[i] not in recommen_list:
                    recommen_list.append(index_loc_of_related_descriptions[i])
                count = count + 1

        print("Recommendations=")
        print(recommen_list)



        #Recommendations
        list_recommendations=[]
        for i in recommen_list:
            print "###################Prediction",j+1,"#####################"
            print movie_data_final.loc[i,:].title
            #print movie_data_final.loc[i,:].plot_movie
            j=j+1
            movie={}
            movie["movie_id"]=movie_data_final.loc[i,:].movie_id
            movie["title"]=movie_data_final.loc[i,:].title
            movie["plot_movie"]=movie_data_final.loc[i,:].plot_movie
            movie["genres"]=str(movie_data_final.loc[i,:][3])+","+str(movie_data_final.loc[i,:][4])+","+str(movie_data_final.loc[i,:][5])
            movie["rating"]=movie_data_final.loc[i,:].rating
            movie["director"]=movie_data_final.loc[i,:].director
            movie["poster"]=movie_data_final.loc[i,:].poster
            movie["languages"]=movie_data_final.loc[i,:].languages
            movie["year"]=movie_data_final.loc[i,:].year
            movie["votes"]=movie_data_final.loc[i,:].votes
            movie["writer"]=movie_data_final.loc[i,:].writer
            movie["awards"]=str(movie_data_final.loc[i,:].awards)
            movie["runtime"]=movie_data_final.loc[i,:].runtime
            movie["country"]=movie_data_final.loc[i,:].country
            movie["released"]=movie_data_final.loc[i,:].released
            movie["languages"]=movie_data_final.loc[i,:].languages
            movie["production"]=movie_data_final.loc[i,:].production

            # movie["plot_movie"]=movie_data_final.loc[i,:].plot_movie
            list_recommendations.append(movie)
        print "##################Completed###########################"
        return list_recommendations

# def get_genre_book()

def get_movie_recommendation(movie_id,movie_data_final,count_vect_books,X_train_tfidf_books,tfidf_transformer_books,clf,book_info):
    #Movie input book output
    input=movie_data_final.loc[movie_id-1,:].plot_movie
    input_title = movie_data_final.loc[movie_id-1, :].title
    input = [input]
    X_new_counts = count_vect_books.transform(input)

    #Naive Bayer preiction
    X_new_tfidf_books = tfidf_transformer_books.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf_books)


    ###############CHECK SIMILARITY#####################
    check_similarity_with_each_record = cosine_similarity(X_new_counts, X_train_tfidf_books)
    flattened_sim_scores = check_similarity_with_each_record.flatten()
    index_loc_of_related_descriptions = np.argsort(flattened_sim_scores)[::-1][:10]
    Confidence_values = flattened_sim_scores[index_loc_of_related_descriptions]

    j = 0
    prediction_df = pd.DataFrame()
    list_recommendations=[]
    for i in index_loc_of_related_descriptions:
        #check if it contains 
        book={}
        book["book_id"]=book_info.loc[i, :].book_id
        book["book_title"]=book_info.loc[i, :].title
        book["description"]=book_info.loc[i, :].Description
        book["rating"]=book_info.loc[i, :].Rating
        # book["genres"]='missing' if x is np.nan else x for x in (book_info.loc[i, :].Genre_Name_Book_Final)]
        if(pd.isnull(book_info.loc[i, :].Genre_Name_Book_Final)):
            book["genres"]="missing"
        else:
            book["genres"]=book_info.loc[i, :].Genre_Name_Book_Final
        book["Author_names"]=book_info.loc[i, :].Author_names
        book["Url"]=book_info.loc[i, :].url
        book["Publisher"]=book_info.loc[i, :].Publisher
        book["isbn_13"]=book_info.loc[i, :].isbn_13
        book["Subjects"]=book_info.loc[i, :].Subjects
        book["Publish_date"]=book_info.loc[i, :].Publish_date



        list_recommendations.append(book)
        prediction_df=prediction_df.append(book_info.loc[i,:],ignore_index=True)

        j=j+1
    print "##################Completed###########################"
    return list_recommendations


def get_movie_recommendation_user(user_genres,movie_id,movie_data_final,count_vect_books,X_train_tfidf_books,tfidf_transformer_books,clf,book_info):
    #Movie input book output
    input=movie_data_final.loc[movie_id-1,:].plot_movie
    input_title = movie_data_final.loc[movie_id-1, :].title
    input = [input]
    X_new_counts = count_vect_books.transform(input)

    #Naive Bayer preiction
    X_new_tfidf_books = tfidf_transformer_books.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf_books)


    ###############CHECK SIMILARITY#####################
    check_similarity_with_each_record = cosine_similarity(X_new_counts, X_train_tfidf_books)
    flattened_sim_scores = check_similarity_with_each_record.flatten()
    index_loc_of_related_descriptions = np.argsort(flattened_sim_scores)[::-1][:10]
    Confidence_values = flattened_sim_scores[index_loc_of_related_descriptions]

    #get genres of user
    list_genres = user_genres.split(",")
    list_genres = filter(None, list_genres)

    recommen_list = []
    count = 0
    z = 0
    for i in index_loc_of_related_descriptions:
        if str(book_info.loc[i,:].Genre_Name_Book_Final) in list_genres:
            if count < 5:
                recommen_list.append(index_loc_of_related_descriptions[z])
                count = count + 1
                z = z + 1
            else:
                break
        else:
            z = z + 1

    for i in range(0,10):
        if count >=5:
            break
        else:
            if index_loc_of_related_descriptions[i] not in recommen_list:
                recommen_list.append(index_loc_of_related_descriptions[i])
            count = count + 1

    j = 0
    list_recommendations=[]
    for i in recommen_list:
        #check if it contains 
        book={}
        book["book_id"]=book_info.loc[i, :].book_id
        book["book_title"]=book_info.loc[i, :].title
        book["description"]=book_info.loc[i, :].Description
        book["rating"]=book_info.loc[i, :].Rating
        # book["genres"]='missing' if x is np.nan else x for x in (book_info.loc[i, :].Genre_Name_Book_Final)]
        if(pd.isnull(book_info.loc[i, :].Genre_Name_Book_Final)):
            book["genres"]="missing"
        else:
            book["genres"]=book_info.loc[i, :].Genre_Name_Book_Final
        book["Author_names"]=book_info.loc[i, :].Author_names
        book["Url"]=book_info.loc[i, :].url
        book["Publisher"]=book_info.loc[i, :].Publisher
        book["isbn_13"]=book_info.loc[i, :].isbn_13
        book["Subjects"]=book_info.loc[i, :].Subjects
        book["Publish_date"]=book_info.loc[i, :].Publish_date



        list_recommendations.append(book)

        j=j+1
    print "##################Completed###########################"
    return list_recommendations

def get_grenre_books(list_books,book_info):
    gen_list=""
    for book in list_books:
        book=str(book)
        gen = book_info.loc[book_info['title'] == book].Genre_Name_Book_Final
        if np.any(gen.values):
            print((gen.values[0]))
            gen_list=gen_list+str(gen.values[0])+","
    return(gen_list)

def get_grenre_movies(list_movies,movie_data_final):
    gen_list=""
    for movie in list_movies:
        movie=str(movie)
        gen = movie_data_final.loc[movie_data_final['title'] == movie]
        if np.any(gen[0].values):
            gen_list=gen_list+(str(gen[0].values[0])+","+str(gen[1].values[0])+","+str(gen[2].values[0])+",")
    return(gen_list)


data=setup()
get_grenre_movies(["100 Girls","10 Rillington Place"],data["movie_data_final"])
# get_movie_recommendation(1,data["movie_data_final"],data["count_vect_books"],data["X_train_tfidf_books"],data["tfidf_transformer_books"],data["clf"],data["book_info"])
