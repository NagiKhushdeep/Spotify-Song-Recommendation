import pandas as pd
import numpy as np
import pickle

from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import streamlit as st

import re

loaded_model = pickle.load(open('spotify.pkl', 'rb'))

#####################################Recommendations####################################################
dataf=pd.read_csv('recommendations.csv',header=0)
df = pd.read_csv('spotify_songs_w_genre.csv',header=0)
df['consolidates_genre_lists']=df['consolidates_genre_lists'].astype(str)
df['genres_upd'] = df['consolidates_genre_lists'].apply(lambda x: re.findall(r"'([^']*)'", x))

def new_func(dataf):
    x=dataf[dataf.drop(columns=['artists','name']).columns].values
    scaler =StandardScaler().fit(x)
    return x,scaler

x, scaler = new_func(dataf)
X_scaled = scaler.transform(x)
dataf[dataf.drop(columns=['artists','name']).columns]=X_scaled
dataf.sort_values('tempo',inplace=True)

def find_song(song_name, df, number=10):
    song_names = df['name'].values
    artists = df['artists'].values
    song_list = []
    count = 0

    #if song_name[-1] == ' ':
    #    song_name = song_name[:-1]
    for i in song_names:
        if song_name.lower() in i.lower():
            song_list.append([len(song_name) / len(i), count])
        else:
            song_list.append([0, count])
        count += 1

    song_list.sort(reverse=True)  # list containing list of len(song_name)/len(i) value and count (row number)
    s = [[song_names[song_list[i][1]], artists[song_list[i][1]].strip('][').split(', ')] for i in
         range(number)]  # list containing list of song name and its artist name/names
    songs = [song_names[song_list[i][1]] for i in range(number)]  # list containing just the song names
    artist = [artists[song_list[i][1]] for i in range(number)]  # list containing just the artist names

    x = []
    for i in s:
        l = ''
        by = ''
        for j in i[1]:
            by += j
        l += i[0] + ' by ' + by
        x.append(l)  # list of strings containing song names and artists in the form "song_name by artist_name"

    slist = []  # this will be a list containing tuples of songs with name equal to what user has entered
    for i in range(number):
        slist.append((x[i], i))  # appending song and its artists with an index as a tuple

    return slist, songs, artist


def find_cos_dist(df, song, number, artist, st):
    x = df[(df['name'] == song) & (df['artists'] == artist)].drop(
        columns=['name', 'artists']).values  # vector for the user entered song

    artist = artist.replace("'", "").replace("'", "").replace('[', '').replace(']', '')
    if ',' in artist:
        inm = artist.rfind(",")
        artist = artist[:inm] + ' and' + artist[inm + 1:]

    song_names = df['name'].values
    p = []
    count = 0

    for i in df.drop(columns=['artists', 'name']).values:
        p.append([distance.cosine(x, i), count])
        count += 1
    p.sort()  # list of all cosine distances with row count
    st.header('The songs closest to your search ' + song + ' by ' + artist + ' :')
    for i in range(1, number + 1):
        artists = dataf['artists'].values
        artist = artists[p[i][1]]
        artist = artist.replace("'", "").replace("'", "").replace('[', '').replace(']', '')
        if ',' in artist:  # dealing with multiple artists
            inm = artist.rfind(",")
            artist = artist[:inm] + ' and' + artist[inm + 1:]
        st.subheader(song_names[p[i][1]] + ' - '+ artist)

def find_genre(genre, df, number=10):

    genre_names = df['genres_upd'].values
    genre_list=genre_names.tolist()
    flatten_list = [item for subl in genre_list for item in subl]
    genre_set=set(flatten_list)
    genre_list=list(genre_set)
    g_list = []
    count = 0
    
    for i in genre_set:
        if genre.lower() in i.lower():
            g_list.append([len(genre) / len(i), count])
        else:
            g_list.append([0, count])
        count += 1

    g_list.sort(reverse=True)
    
    g = [[genre_list[g_list[i][1]]] for i in range(number)]  # list containing list of song name and its artist name/names
    final_list = [item for subl in g for item in subl]
    return final_list

def genre_song_recom(genre,n):
    new=dict()
    for i,data in df.iterrows():
        if genre in data['genres_upd']:
            new[df.at[i,'name']]=df.at[i,'popularity']
    p = sorted(new.items(), key=lambda x: x[1],reverse=True)
    st.header('The songs belonging to the '+ genre +' are:')
    for i in p[0:n]:
        st.subheader(i[0]) 

def spotify_show():      ## UI for user input uses streamlit
    

    st.header("Predict Popularity of a song")
    year = st.slider("Year", min_value=1921, max_value=2020)
    st.text("Year in which the song was released")
    danceability = st.slider("Danceability", min_value=0.0, max_value=1.0)
    st.text('One a scale of 0-1, how much suitable the song is to dance')
    energy = st.slider("Energy", min_value=0.0, max_value=1.0)
    st.text('One a scale of 0-1, what is the intensity level of the song')
    artists_popularities = st.slider("Artist Popularity", min_value=0.0, max_value=100.0)
    st.text('One a scale of 0-1, what is the popularity level of the artist')
    loudness = st.number_input("Loudness", min_value=-100.0, max_value=50.0)
    st.text('One a scale of -100 to 50, what is the loudness level of the song')
    tempo = st.number_input("Tempo", min_value=0.0, max_value=250.0)
    st.text('One a scale of 0-250, what is the tempo(in beats per minute) of the song')
    predict = st.button("Predict Popularity")

    if predict:
        test_data = np.array([[year, danceability, energy, loudness, tempo, artists_popularities]])
        output = loaded_model.predict(test_data)[0]
        st.subheader('Popularity score of the song (out of 100):')
        st.success(output)

    st.header("Get recommendations by song")
    song_name = st.text_input("Enter the name of the song you like")
    no_of_recom = st.slider("The number of recommendations you want", 1, 20,key="slider1")
    tup, s, ar = find_song(song_name, dataf)
    if song_name:
        st.subheader(f'Closest songs to-> {song_name}:')
        st.text('You can make the song choice from the below dropdown')
        xx = st.selectbox("", options=tup, key='ishu', )
        if xx:
            find_cos_dist(dataf, s[xx[1]], no_of_recom, ar[xx[1]], st)
    
    st.header("Get recommendations by genre")
    genre_name = st.text_input("Enter the genre you like")
    no_of_rec= st.slider("The number of recommendations you want", 1, 20,key="slider2")
    tup2=find_genre(genre_name,df)
    if genre_name:
        st.subheader(f'Songs from-> {genre_name}:')
        st.text('You can make the song choice from the below dropdown')
        yy = st.selectbox("", options=tup2, key='khush', )
        if yy:
            genre_song_recom(yy,no_of_rec)




spotify_show()














