import streamlit as st
import pandas as pd
import pickle
import zipfile
from datetime import datetime
startTime = datetime.now()

zf = zipfile.ZipFile('model_song_names.sv.zip', 'r', zipfile.ZIP_DEFLATED)
model = pickle.load(zf.open('model_song_names.sv'))
song_data = pd.read_csv("song_data.csv")

song_data.drop_duplicates()
song_data.rename(columns={"song_name": "Label", "song_popularity": "Rate"}, inplace=True)
cols = ["Label", "Rate"]
song_labels = song_data[cols].copy()

audio_m = {0:"No", 1:"Yes"}
gen = (i for i in range(12))
dur_val, aco_val, dan_val, en_val, ins_val, tem_val, key_val, liv_val, lou_val, mod_val, spe_val, va_val = gen

def main():

    st.set_page_config(page_title="Popular Song App", menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })
    overview = st.container()
    left, right = st.columns(2)
    prediction, labels = st.columns(2)
    footer = st.container()

    with overview:
        st.title("Song Popularity Prediction App")
        options = st.selectbox('Search song...', song_labels['Label'])
        if options != '':
            sd = song_data.loc[song_data['Label'] == options]
            dur_val = int(sd.iloc[0]['song_duration_ms'])
            aco_val = float(sd.iloc[0]['acousticness'])
            dan_val = float(sd.iloc[0]['danceability'])
            en_val = float(sd.iloc[0]['energy'])
            ins_val = float(sd.iloc[0]['instrumentalness'])
            tem_val = int(sd.iloc[0]['tempo'])
            key_val = int(sd.iloc[0]['key'])
            liv_val = float(sd.iloc[0]['liveness'])
            lou_val = float(sd.iloc[0]['loudness'])
            mod_val = int(sd.iloc[0]['audio_mode'])
            spe_val = float(sd.iloc[0]['speechiness'])
            va_val = float(sd.iloc[0]['audio_valence'])

    with left:
        duration = st.slider( "Duration msec", value=dur_val, min_value=0, max_value=1799346)
        acoustics = st.slider( "Acoustics", value=aco_val, min_value=0.000, max_value=0.999, step=0.001)
        dance = st.slider( "Dance ability", value=dan_val,  min_value=0.000, max_value=0.999, step=0.001)
        energy = st.slider( "Energy",value=en_val,  min_value=0.000, max_value=0.999, step=0.001)
        instrumentals = st.slider( "Instrumentals", value=ins_val,  min_value=0.000, max_value=0.999, step=0.001)
        tempo = st.slider( "Tempo", value=tem_val, min_value=0, max_value=250)

    with right:
        key = st.slider("Key", value=key_val, min_value=0, max_value=11)
        liveness = st.slider("Liveness", value=liv_val, min_value=0.000, max_value=0.999, step=0.001)
        loudness = st.slider("Loudness", value=lou_val, min_value=-40.000, max_value=1.585, step=0.001)
        audio_mode = st.radio("Audio mode", list(audio_m.keys()), index=mod_val, format_func=lambda x: audio_m[x])
        speechiness = st.slider("Speechiness", value=spe_val, min_value=0.000, max_value=0.999, step=0.001)
        audio_valence = st.slider("Audio valence", value=va_val, min_value=0.000, max_value=0.999, step=0.001)



    data = [[duration, acoustics, dance, energy, instrumentals, key, liveness, loudness, audio_mode, speechiness, tempo, audio_valence]]
    popular = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Rate prediction:")
        st.markdown(("Yes, these options are eligible for popular song. Rate is " + str(popular[0]) if popular[0] > 75 else "No, song with these options may not be popular. Rate is below 75"))
        st.text("Current rate is " + str(popular[0]))
        # data.iloc[data['song_popularity'] == popular[0]]
        st.write("Prob {0:.2f} %".format(s_confidence[0][popular][0] * 100))
        st.image("https://raw.githubusercontent.com/Masterx-AI/Project_Song_Popularity_Prediction_/main/songs.jpg")

    with labels:
        st.subheader("Songs with such rate:")
        st.dataframe(song_labels.loc[song_labels['Rate'] == popular[0]])

    with footer:
        st.subheader("Reference to dataset: [link](https://www.kaggle.com/datasets/yasserh/song-popularity-dataset?resource=download)")

if __name__ == "__main__":
    main()


#%%
