import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
from plotly.subplots import make_subplots



# Function to add a footer
def add_footer():
    footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>© 2024 Udinus Center of Excellence. All rights reserved.</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

# Fungsi untuk menampilkan plot menggunakan Plotly di Streamlit
def sampah_tahunan(df):
    # Kelompokkan data berdasarkan tahun
    grouped_data = df.groupby("Tahun").sum().reset_index()

    # Buat plot dengan menggunakan Plotly
    fig = go.Figure()

    # Tambahkan trace untuk masing-masing kategori sampah dengan jenis bar
    fig.add_trace(go.Bar(x=grouped_data["Tahun"], y=grouped_data["Sampah Daun"], name='Sampah Daun', marker_color='#1f77b4'))  # blue
    fig.add_trace(go.Bar(x=grouped_data["Tahun"], y=grouped_data["Sampah Sayuran"], name='Sampah Sayuran', marker_color='#ff7f0e'))  # orange
    fig.add_trace(go.Bar(x=grouped_data["Tahun"], y=grouped_data["Sampah Fermentasi"], name='Sampah Fermentasi', marker_color='#2ca02c'))  # green
    fig.add_trace(go.Bar(x=grouped_data["Tahun"], y=grouped_data["Daun Terolah"], name='Daun Terolah', marker_color='#9467bd'))  # purple
    fig.add_trace(go.Bar(x=grouped_data["Tahun"], y=grouped_data["Kompos Jadi"], name='Kompos Jadi', marker_color='#17becf'))  # cyan

    # Update layout
    fig.update_layout(
        title='Perbandingan Sampah Daun, Sampah Sayuran, Sampah Fermentasi, Daun Terolah, dan Kompos Jadi per Tahun',
        xaxis_title='Tahun',
        yaxis_title='Jumlah Sampah',
        barmode='group',  # Mengelompokkan bar untuk setiap tahun
        legend_title='Kategori Sampah',
        font=dict(size=12, color='black'),  # Ukuran dan warna font
    )

    # Tampilkan plot menggunakan st.plotly_chart
    st.plotly_chart(fig)

def sampah_bulanan(df):
    # Convert 'Tahun' and 'Bulan' columns to a single datetime column
    df['Date'] = pd.to_datetime(df['Tahun'].astype(str) + '-' + df['Bulan'])

    # Plotting with Plotly
    fig = go.Figure()

    # Add traces for each type of waste
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Sampah Daun'], mode='lines+markers', name='Sampah Daun'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Sampah Sayuran'], mode='lines+markers', name='Sampah Sayuran'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Sampah Anorganik'], mode='lines+markers', name='Sampah Anorganik'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Daun Terolah'], mode='lines+markers', name='Daun Terolah'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Sampah Fermentasi'], mode='lines+markers', name='Sampah Fermentasi'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Kompos Jadi'], mode='lines+markers', name='Kompos Jadi'))

    # Update layout
    fig.update_layout(
        title='Perbandingan Sampah per Bulan',
        xaxis_title='Date',
        yaxis_title='Jumlah Sampah',
        legend=dict(x=0, y=1),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    # Tampilkan plot menggunakan st.plotly_chart
    st.plotly_chart(fig)



def forecast(df, model, feature):
    df['Date'] = pd.to_datetime(df[['Tahun', 'Bulan']].assign(DAY=1).astype(str).agg('-'.join, axis=1))
    df.set_index('Date', inplace=True)
    ts_data = df[feature].dropna()

    # Load the model from the file
    with open(model, 'rb') as file:
        loaded_model = pickle.load(file)

    # Assuming ts_data and df are defined previously
    # ts_data: the original time series data
    # df: a DataFrame that includes the in-sample forecast

    # Calculate in-sample forecast
    df['forecast_in_sample'] = loaded_model.predict_in_sample()

    # Calculate out-of-sample forecast
    forecast_out_sample = loaded_model.predict(n_periods=12)  # Forecast for the next 12 months

    # Create the index for the out-sample forecast
    out_sample_index = pd.date_range(start=ts_data.index[-1], periods=13, freq='M')[1:]

    # Combine in-sample and out-sample forecasts
    combined_forecast = pd.concat([df['forecast_in_sample'], pd.Series(forecast_out_sample, index=out_sample_index)])

    # Plotting with Plotly
    fig = make_subplots()

    # Original time series data
    fig.add_trace(go.Scatter(
        x=ts_data.index,
        y=ts_data,
        mode='lines+markers',
        name='Original',
        line=dict(color='blue'),
        marker=dict(symbol='circle')
    ))

    # In-sample forecast
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['forecast_in_sample'],
        mode='lines+markers',
        name='Forecast (In-sample)',
        line=dict(color='green'),
        marker=dict(symbol='circle')
    ))

    # Out-of-sample forecast
    fig.add_trace(go.Scatter(
        x=out_sample_index,
        y=forecast_out_sample,
        mode='lines+markers',
        name='Forecast (Out-sample)',
        line=dict(color='green'),
        marker=dict(symbol='circle')
    ))

    # Adding a connection between the last in-sample point and the first out-sample point
    fig.add_trace(go.Scatter(
        x=[df.index[-1], out_sample_index[0]],
        y=[df['forecast_in_sample'].iloc[-1], forecast_out_sample[0]],
        mode='lines+markers',
        line=dict(color='green'),
        marker=dict(symbol='circle'),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f'Time Series {feature} Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
        legend=dict(x=0, y=1),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    # Tampilkan plot menggunakan st.plotly_chart
    st.plotly_chart(fig)

# Function to create and display pie chart for a location
def create_pie_chart(df, location):
    # Filter data for the selected location
    df_location = df[df['Lokasi'] == location]

    # Sum up waste categories
    total_sampah_daun = df_location['Sampah Daun'].sum()
    total_sampah_sayuran = df_location['Sampah Sayuran'].sum()
    total_sampah_fermentasi = df_location['Sampah Fermentasi'].sum()
    total_daun_terolah = df_location['Daun Terolah'].sum()
    total_kompos_jadi = df_location['Kompos Jadi'].sum()

    # Create pie chart
    fig = go.Figure()

    # Add trace for each waste category
    fig.add_trace(go.Pie(labels=['Sampah Daun', 'Sampah Sayuran', 'Sampah Fermentasi', 'Daun Terolah', 'Kompos Jadi'],
                         values=[total_sampah_daun, total_sampah_sayuran, total_sampah_fermentasi, total_daun_terolah, total_kompos_jadi],
                         hole=0.3,
                         marker=dict(line=dict(color='#000000', width=1))
                         ))

    # Update layout
    fig.update_layout(
        title=f'Komposisi Sampah di Lokasi {location}',
        legend_title='Kategori Sampah',
        uniformtext_minsize=12, uniformtext_mode='hide'
    )

    # Display pie chart
    st.plotly_chart(fig)

# Function to load data from an Excel file
def load_data(file_path):
    try:
        df = pd.read_excel(file_path, index_col=0)
        st.write("Data loaded successfully")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to save data to an Excel file
def save_data(df, file_path):
    try:
        df.to_excel(file_path, index=False)
        st.write("Data saved successfully")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Error saving data: {e}")

# Function to add a new record
def add_record(df, record):
    # Convert record into a list of dictionaries
    records_list = [record]

    # Convert list of dictionaries into a DataFrame
    df_baru = pd.DataFrame(records_list)

    # Concatenate new DataFrame with existing DataFrame
    df = pd.concat([df, df_baru], ignore_index=True)
    return df

# Function to update a record
def update_record(df, record, record_id):
    if record_id in df.index:
        df.loc[record_id] = record
    return df

# Function to delete a record
def delete_record(df, record_id):
    return df.drop(record_id, errors='ignore')

# Main function
def main():
    # Set page configuration
    st.set_page_config(layout="wide")

    

    # setting sidebar
    option = st.sidebar.selectbox("Operation", ["Dashboard", "Input Data"])

    if option == "Dashboard":
        # Styling for header
        st.markdown("""
            <style>
            .header {
                color: white;
                padding: 10px 0;
                text-align: center;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Header section
        st.markdown('<div class="header"><h1>Dashboard</h1></div>', unsafe_allow_html=True)

        col11, col12 = st.columns(2)
        
        with col11:
            # Dataset
            df_bulanan = pd.read_excel("Data/Laporan-hasil-Rumah-Kompos-gabungan.xlsx")
            
            # Konversi kolom Tahun ke tipe data datetime
            df_bulanan['Tahun'] = pd.to_datetime(df_bulanan['Tahun'], format='%Y')

            # Tampilkan hanya tahun saja
            df_bulanan['Tahun'] = df_bulanan['Tahun'].dt.year

            # Multiselect untuk memilih tahun
            selected_years = st.multiselect('Pilih Tahun', df_bulanan['Tahun'].unique())

            # Multiselect untuk memilih bulan
            selected_months = st.multiselect('Pilih Bulan', df_bulanan['Bulan'].unique())

            # Filter DataFrame berdasarkan pilihan pengguna
            if selected_years and not selected_months:
                # Filter hanya berdasarkan tahun yang dipilih
                df_bulanan_filtered = df_bulanan[df_bulanan['Tahun'].isin(selected_years)]
            elif selected_months and not selected_years:
                # Filter hanya berdasarkan bulan yang dipilih
                df_bulanan_filtered = df_bulanan[df_bulanan['Bulan'].isin(selected_months)]
            elif selected_years and selected_months:
                # Filter berdasarkan tahun dan bulan yang dipilih
                mask = (df_bulanan['Tahun'].isin(selected_years)) & (df_bulanan['Bulan'].isin(selected_months))
                df_bulanan_filtered = df_bulanan[mask]
            else:
                # Jika tidak ada pilihan, tampilkan DataFrame secara keseluruhan
                df_bulanan_filtered = df_bulanan

            # Tampilkan DataFrame hasil filter
            st.dataframe(df_bulanan_filtered)

            # Button for downloading CSV
            st.download_button(
                label="Download data as CSV",
                data= convert_df(df_bulanan_filtered),
                file_name="data.csv",
                mime="text/csv",
            )
        
        with col12:
            # Dataset
            df_harian = pd.read_csv("Data/Laporan hasil Rumah Kompos gabungan harian.csv")

            # Ubah kolom 'Tanggal' menjadi datetime
            df_harian['Tanggal'] = pd.to_datetime(df_harian['Tanggal'])

            # Hapus kolom 'Bulan' dan 'Tahun'
            df_harian.drop(columns=['Bulan', 'Tahun'], inplace=True)

            # Ubah nama kolom 'Kompos' menjadi 'Lokasi'
            df_harian.rename(columns={'Kompos': 'Lokasi'}, inplace=True)

            # Ganti nilai 'kompos pasar kendal' menjadi 'Pasar Kendal' dan 'kompos jatirejo' menjadi 'Jatirejo'
            df_harian['Lokasi'] = df_harian['Lokasi'].replace({'Kompos Pasar Kendal': 'Pasar Kendal', 'Kompos Jatirejo': 'Jatirejo'})

        # Input untuk memilih tanggal
            min_date = df_harian['Tanggal'].min().date() if not df_harian.empty else pd.Timestamp.min.date()
            max_date = df_harian['Tanggal'].max().date() if not df_harian.empty else pd.Timestamp.max.date()

            selected_date = st.date_input('Pilih Tanggal', min_value=min_date, max_value=max_date, value=None)


            # Multiselect untuk memilih lokasi
            selected_locations = st.multiselect('Pilih Lokasi', df_harian['Lokasi'].unique())

            # Filter DataFrame berdasarkan pilihan pengguna
            if selected_date and not selected_locations:
                # Filter hanya berdasarkan tanggal yang dipilih
                df_harian_filtered = df_harian[df_harian['Tanggal'].dt.date == selected_date]
            elif selected_locations and not selected_date:
                # Filter hanya berdasarkan lokasi yang dipilih
                df_harian_filtered = df_harian[df_harian['Lokasi'].isin(selected_locations)]
            elif selected_date and selected_locations:
                # Filter berdasarkan tanggal dan lokasi yang dipilih
                mask = (df_harian['Tanggal'].dt.date == selected_date) & (df_harian['Lokasi'].isin(selected_locations))
                df_harian_filtered = df_harian[mask]
            else:
                # Jika tidak ada pilihan, tampilkan DataFrame secara keseluruhan
                df_harian_filtered = df_harian

            # Tampilkan DataFrame hasil filter
            st.dataframe(df_harian_filtered)


            # Button for downloading CSV
            st.download_button(
                label="Download data as CSV",
                data=convert_df(df_harian_filtered),
                file_name="filtered_data.csv",
                mime="text/csv",
            )
        
    # PLOT
        col21, col22 = st.columns(2)
        with col21:
            sampah_tahunan(df_bulanan_filtered)
        with col22:
            sampah_bulanan(df_bulanan_filtered)

        col31, col32 = st.columns(2)
        with col31:
            create_pie_chart(df_harian_filtered, "Pasar Kendal")
        with col32:
            create_pie_chart(df_harian_filtered, "Jatirejo")
        
        col41, col42 = st.columns(2)
        
        with col41:
            forecast(df_bulanan, "Model/model_sampahDaun.pkl", "Sampah Daun")
        with col42:
            forecast(df_bulanan, "Model/model_sampahSayuran.pkl", "Sampah Sayuran")
        
        col51, col52 = st.columns(2)

        with col51:
            forecast(df_bulanan, "Model/model_daunTerolah.pkl", "Daun Terolah")
        with col52:
            forecast(df_bulanan, "Model/model_sampahFermentasi.pkl", "Sampah Fermentasi")
        
        forecast(df_bulanan, "Model/model_komposJadi.pkl", "Kompos Jadi")
    
    elif option == "Input Data":
        # Styling for header
        st.markdown("""
            <style>
            .header {
                color: white;
                padding: 10px 0;
                text-align: center;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Header section
        st.markdown('<div class="header"><h1>Input Data</h1></div>', unsafe_allow_html=True)

        # Dataset
        df_bulanan = pd.read_excel("Data/Laporan-hasil-Rumah-Kompos-gabungan.xlsx")
        
        # Konversi kolom Tahun ke tipe data datetime
        df_bulanan['Tahun'] = pd.to_datetime(df_bulanan['Tahun'], format='%Y')

        # Tampilkan hanya tahun saja
        df_bulanan['Tahun'] = df_bulanan['Tahun'].dt.year

        st.dataframe(df_bulanan)

        tab1, tab2, tab3 = st.tabs(["Add Data" , "Update Data", "Delete Data"])

        with tab1:
            st.subheader("Add New Record")
            tahun = st.number_input("Tahun", min_value=2000, max_value=2100)
            bulan_add = st.selectbox("Bulan", [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ])
            sampah_daun = st.number_input("Sampah Daun", min_value=0)
            sampah_sayuran = st.number_input("Sampah Sayuran", min_value=0)
            sampah_anorganik = st.number_input("Sampah Anorganik", min_value=0)
            daun_terolah = st.number_input("Daun Terolah", min_value=0)
            sampah_fermentasi = st.number_input("Sampah Fermentasi", min_value=0)
            kompos_jadi = st.number_input("Kompos Jadi", min_value=0)
            if st.button("Add"):
                new_record = {
                    'Tahun': tahun,
                    'Bulan': bulan_add,
                    'Sampah Daun': sampah_daun,
                    'Sampah Sayuran': sampah_sayuran,
                    'Sampah Anorganik': sampah_anorganik,
                    'Daun Terolah': daun_terolah,
                    'Sampah Fermentasi': sampah_fermentasi,
                    'Kompos Jadi': kompos_jadi
                }
                df_bulanan = add_record(df_bulanan, new_record)
                save_data(df_bulanan, "Data/Laporan-hasil-Rumah-Kompos-gabungan.xlsx")
                st.success("Record added successfully")
                
        with tab2:
            st.subheader("Update Record")
            record_id = st.number_input("Enter ID of the record to update", min_value=0)
            if record_id in df_bulanan.index:
                tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=int(df_bulanan.loc[record_id, 'Tahun']))
                bulan_update = st.selectbox("Bulan", [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
            ], index=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"].index(df_bulanan.loc[record_id, 'Bulan']), key='update')
                sampah_daun = st.number_input("Sampah Daun", min_value=0, value=int(df_bulanan.loc[record_id, 'Sampah Daun']))
                sampah_sayuran = st.number_input("Sampah Sayuran", min_value=0, value=int(df_bulanan.loc[record_id, 'Sampah Sayuran']))
                sampah_anorganik = st.number_input("Sampah Anorganik", min_value=0, value=int(df_bulanan.loc[record_id, 'Sampah Anorganik']))
                daun_terolah = st.number_input("Daun Terolah", min_value=0, value=int(df_bulanan.loc[record_id, 'Daun Terolah']))
                sampah_fermentasi = st.number_input("Sampah Fermentasi", min_value=0, value=int(df_bulanan.loc[record_id, 'Sampah Fermentasi']))
                kompos_jadi = st.number_input("Kompos Jadi", min_value=0, value=int(df_bulanan.loc[record_id, 'Kompos Jadi']))
                if st.button("Update"):
                    updated_record = {
                        'Tahun': tahun,
                        'Bulan': bulan_update,
                        'Sampah Daun': sampah_daun,
                        'Sampah Sayuran': sampah_sayuran,
                        'Sampah Anorganik': sampah_anorganik,
                        'Daun Terolah': daun_terolah,
                        'Sampah Fermentasi': sampah_fermentasi,
                        'Kompos Jadi': kompos_jadi
                    }
                    df_bulanan = update_record(df_bulanan, updated_record, record_id)
                    save_data(df_bulanan,  "Data/Laporan-hasil-Rumah-Kompos-gabungan.xlsx")
                    st.success("Record updated successfully")
            else:
                st.error("Record not found")
        with tab3:
            st.subheader("Delete Record")
            record_id = st.number_input("Enter ID of the record to delete", min_value=0)
            if record_id in df_bulanan.index:
                if st.button("Delete"):
                    df_bulanan = delete_record(df_bulanan, record_id)
                    save_data(df_bulanan, "Data/Laporan-hasil-Rumah-Kompos-gabungan.xlsx")
                    st.success("Record deleted successfully")
                    st.experimental_rerun()
            else:
                st.error("Record not found")
        

    

    # Add footer
    add_footer()

# Run the main function
if __name__ == "__main__":
    main()
