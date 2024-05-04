from pycaret.regression import *
import streamlit as st
import pandas as pd
import numpy as np


# Load trained machine learning model
model = load_model('property_pipeline')

dataset = pd.read_csv("processed_data.csv")
dataset = dataset.iloc[:2000]

def predict(model, input_df):
    prediction_df = predict_model(model, data = input_df)
    st.write(prediction_df)
    predictions= prediction_df['prediction_label'][0]
    return predictions

def run():
    from PIL import Image
    image = Image.open('images/investasi-properti.png')
    image_property = Image.open('images/gambar1.jpg')

    st.sidebar.title('Praktikum Streamlit')
    st.sidebar.markdown('Aplikasi Prediksi Harga Properti di Indonesia')
    st.sidebar.image(image)
    st.sidebar.info('Isi form dibawah untuk mengetahui perkiraan harga properti dengan kriteria yang diinginkan.')

    # List provinsi dan kota
    provinsi_list = [
        'Jawa Timur', 'Jakarta', 'Jawa Barat', 'Banten', 'Bali', 
        'Jawa Tengah', 'Yogyakarta', 'Kepulauan Riau', 'Sulawesi Selatan', 
        'Sumatera Utara', 'Kalimantan Timur', 'Riau', 'Lampung', 
        'Sumatera Selatan', 'Sulawesi Utara'
    ]

    # Isi dengan daftar provinsi yang ada
    kota_dict = {
        'Jawa Timur': ['Surabaya', 'Sidoarjo', 'Malang', 'Gresik', 'Mojokerto', 'Jember', 'Pasuruan', 'Batu'],
        'Jakarta': ['Jakarta Selatan', 'Jakarta Barat', 'Jakarta Utara', 'Jakarta Timur', 'Jakarta Pusat'],
        'Jawa Barat': ['Bandung', 'Bekasi', 'Bogor', 'Depok', 'Bandung Barat', 'Cirebon', 'Cimahi', 'Karawang'],
        'Banten': ['Tangerang Selatan', 'Tangerang'],
        'Bali': ['Badung', 'Denpasar', 'Gianyar', 'Tabanan'],
        'Jawa Tengah': ['Semarang', 'Surakarta', 'Karanganyar', 'Sukoharjo', 'Boyolali', 'Demak', 'Klaten'],
        'Yogyakarta': ['Sleman', 'Yogyakarta', 'Gunung Kidul', 'Bantul'],
        'Kepulauan Riau': ['Batam'],
        'Sulawesi Selatan': ['Makassar']
    }

    st.image(image_property)
    st.title('Prediksi Harga Properti di Indonesia')
    st.markdown('Aplikasi ini bertujuan untuk membantu Anda memperkirakan harga properti di berbagai wilayah di Indonesia.')
    st.markdown('Silakan isi formulir di sisi kanan untuk memulai.')

    st.sidebar.header('Masukkan Data Properti')
    buildingSize = st.sidebar.slider('Luas Bangunan (m^2)', min_value=24, max_value= 1000, key='bsize')
    landSize = st.sidebar.slider('Luas Tanah (m^2)', min_value=24, max_value= 1000, key='lsize')
    bedRooms = st.sidebar.number_input('Jumlah Kamar Tidur', min_value=1, key='bed')
    bathRooms = st.sidebar.number_input('Jumlah Kamar Mandi', min_value=1, key='bath')
    garages = st.sidebar.number_input('Jumlah Garasi', min_value=0, key='gar')
    provinsi = st.sidebar.selectbox('Provinsi', provinsi_list, key='prov')
    kota_list = kota_dict[provinsi]
    city = st.sidebar.selectbox('Kota', kota_list, key='city')

    input_df = pd.DataFrame({
            'buildingSize': [buildingSize],
            'landSize': [landSize],
            'bedRooms': [bedRooms],
            'bathRooms': [bathRooms],
            'garages': [garages],
            'province': [provinsi],
            'city': [city]
        })
    

    if st.sidebar.button("Predict"):
        output = predict(model=model, input_df=input_df)
        st.success(f'Harga properti di {city}, {provinsi} dengan kriteria yang diinginkan diperkirakan sebesar: **{output:.2f}**')

    

    st.sidebar.success('by: Mutiara Sanny ☺️')

    st.info('**Tujuan Aplikasi:**\nAplikasi ini bertujuan untuk membantu Anda memperkirakan harga properti di berbagai wilayah di Indonesia dengan menggunakan model machine learning.\n\n**Cara Penggunaan:**\n1. Isi formulir di sisi kanan dengan detail properti yang ingin Anda prediksi.\n2. Klik tombol "Predict" untuk melihat perkiraan harga properti.\n3. Hasil prediksi akan ditampilkan di layar.\n\n**Disclaimer:**\n- Perkiraan harga properti bersifat indikatif dan tidak mengikat. Sebaiknya lakukan pengecekan lebih lanjut atau konsultasi dengan profesional real estate sebelum membuat keputusan pembelian atau penjualan properti.\n- Aplikasi ini bertujuan untuk memberikan panduan awal dalam menentukan harga properti dan bukan sebagai pengganti penilaian profesional oleh ahli real estate.')
    

if __name__ == '__main__':
    run()
