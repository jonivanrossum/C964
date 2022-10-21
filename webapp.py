import streamlit as st
import pickle
import numpy as np
import pandas as pd
from plotly import graph_objs as go

### Code adopted from https://github.com/dataprofessor/code/blob/master/streamlit/part2/iris-ml-app.py
### Login code adopted from https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso#introduction

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if (
            st.session_state["username"] in st.secrets["passwords"]
            and st.session_state["password"]
            == st.secrets["passwords"][st.session_state["username"]]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    st.write("Here goes your normal Streamlit app...")
    st.button("Click me")
    st.write("""
    # Song Popularity Predictor
    ## This app predicts a song's popularity rating and is a tool to discover hidden musical talent!

    Check out the links below for the links to the dataset used and more information about Spotify Audio Features.
    """)

    link = '[Kaggle - Spotify  and Genius Track Dataset](https://www.kaggle.com/datasets/saurabhshahane/spotgen-music-dataset)'
    spotify_api = '[Spotify Audio Features](https://developer.spotify.com/documentation/web-api/reference/#endpoint-get-audio-features)'
    st.markdown(link, unsafe_allow_html=True)
    st.markdown(spotify_api, unsafe_allow_html=True)

    st.sidebar.header('User Input Parameters')


    # Get user inputs
    def user_input_features():
        acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.27)
        danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.74)
        duration_ms = st.sidebar.slider('Duration_ms', 5000, 533800, 192453)
        energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.747)
        instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.0)
        key = st.sidebar.slider('Key', 0, 11, 10)
        mode = st.sidebar.slider('Mode', 0, 1, 1)
        liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.22)
        loudness = st.sidebar.slider('Loudness', -54.0, 3.2, -4.82)
        speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.03)
        tempo = st.sidebar.slider('Tempo', 0.0, 220.0, 106.0)
        time_signature = st.sidebar.slider('Time_Signature', 0, 5, 4)
        valence = st.sidebar.slider('Valence', 0.0, 1.0, 0.48)
        data = {'acousticness': acousticness,
                'danceability': danceability,
                'duration_ms': duration_ms,
                'energy': energy,
                'instrumentalness': instrumentalness,
                'key': key,
                'liveness': liveness,
                'loudness': loudness,
                'mode': mode,
                'speechiness': speechiness,
                'tempo': tempo,
                'time_signature' : time_signature,
                'valence': valence,
                }
        features = pd.DataFrame(data, index=[0])
        return features


    df = user_input_features()

    # Show user inputs
    st.subheader('Song Input Parameters')
    st.write(df)

    # Create Plotly plot
    columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
    df_song_char = df.filter(items=columns)
    y = df_song_char.values.tolist()[0]

    fig = go.Figure(data=go.Bar(x=columns, y=y, marker={'color': y, 'colorscale': 'Viridis'}), layout_title_text='Song Audio Features from User Input')
    st.plotly_chart(fig, use_container_width=True)

    filepath = 'music_pop_predictor.pkl'
    with open(filepath, 'rb') as f:
        model = pickle.load(f);
        
    prediction = model.predict(df)
    p_dict = {1: 'Unpopular', 
             2: 'Popular',
             3: 'Very Popular'}

    st.subheader('PREDICTED SONG POPULARITY')
    prediction = int(np.round(prediction, 0))

    st.write('### A song with these parameters will be:', p_dict[prediction])
