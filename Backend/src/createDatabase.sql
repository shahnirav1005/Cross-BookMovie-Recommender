DROP database IF EXISTS recommender;
CREATE DATABASE recommender;
USE recommender;

CREATE TABLE User(
	username 	  VARCHAR(30),
	password      VARCHAR(30) NOT NULL,
	booksHistory  VARCHAR(400),
	moviesHistory VARCHAR(400),
	booksGenresHistory  VARCHAR(200),
	moviesGenresHistory VARCHAR(200),
	PRIMARY KEY(username)
);

CREATE TABLE Movies(
	movie_id 	VARCHAR(20),
	title		VARCHAR(1500),
	plot_movie	VARCHAR(10000),
	rated		VARCHAR(600),
	genres		VARCHAR(300),
	writer		VARCHAR(500),
	poster		VARCHAR(200),
	rating		FLOAT,
	imdbid		VARCHAR(50),
	year		VARCHAR(4),
	languages	VARCHAR(70),
	actors		VARCHAR(200),
	director	VARCHAR(200)

);

CREATE TABLE Books(
	book_id 		VARCHAR(6),
	title			VARCHAR(350),
	isbn			VARCHAR(13),
	author_names 	VARCHAR(300),
	year			VARCHAR(40),
	genres			VARCHAR(50),
	description		VARCHAR(18500),
	rating			FLOAT,
	url				VARCHAR(200)
);

LOAD DATA LOCAL INFILE 'Book_details.tsv'  
INTO TABLE Books
COLUMNS TERMINATED BY '\t' LINES TERMINATED BY '\n' IGNORE 1 LINES;


LOAD DATA LOCAL INFILE 'Movie_details.tsv'  
INTO TABLE Movies
COLUMNS TERMINATED BY '\t'  LINES TERMINATED BY '\n' IGNORE 1 LINES;