import streamlit as st
import pandas as pd
from urllib.request import urlopen
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import json
import requests
from streamlit_lottie import st_lottie
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import snowflake.connector

#Layout
st.set_page_config(
    page_title="SimiLo",
    layout="centered",
    initial_sidebar_state="expanded")

#Data Pull and Functions
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

@st.cache_data
def pull_clean():
    master_zip=pd.read_csv('MASTER_ZIP.csv',dtype={'ZCTA5': str})
    master_city=pd.read_csv('MASTER_CITY.csv',dtype={'ZCTA5': str})
    return master_zip, master_city



#Options Menu
with st.sidebar:
    selected = option_menu('SimiLo', ["Intro", 'Search','Analyze'], 
        icons=['play-btn','search','info-circle'],menu_icon='intersect', default_index=0)
    lottie = load_lottiefile("Animation.json")
    st_lottie(lottie,key='loc')

#Intro Page
if selected=="Intro":
    #Header
    st.title('Welcome to SimiLo')
    st.subheader('*A new tool to find similar locations across the United States.*')

    st.divider()

    #Use Cases
    with st.container():
        col1,col2=st.columns(2)
        with col1:
            st.header('Use Cases')
            st.markdown(
                """
                - _Remote work got you thinking about relocation?_
                - _Looking for a new vacation spot?_
                - _Conducting market research for product expansion?_
                - _Just here to play and learn?_
                """
                )
        with col2:
            lottie2 = load_lottiefile("place2.json")
            st_lottie(lottie2,key='place',height=300,width=300)

    st.divider()

    #Tutorial Video
    st.header('Tutorial Video')
    video_file = open('Similo_Tutorial3_compressed.mp4', 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    
#Search Page
if selected=="Search":

    st.subheader('Select Location')

    master_zip,master_city=pull_clean()
    master_zip.columns = master_zip.columns.str.upper()
    master_zip = master_zip.rename(columns={'ZCTA5': 'ZIP'})
    master_zip['ZIP'] = master_zip['ZIP'].astype(str).str.zfill(5)
    master_city.columns = master_city.columns.str.upper()

    loc_select=st.radio('Type',['Zip','City'],horizontal=True, label_visibility="collapsed")

    if loc_select=='City':
        city_select=st.selectbox(label='city',options=['City']+list(master_city['CITYSTATE'].unique()),label_visibility='collapsed')
        st.caption('Note: City is aggregated to the USPS designation which may include additional nearby cities/towns/municipalities')
        zip_select='Zip'
    if loc_select=='Zip':
        zip_select = st.selectbox(label='zip',options=['Zip']+list(master_zip['ZIP'].unique()),label_visibility='collapsed')

    with st.expander('Advanced Settings'):

        st.subheader('Filter Results')
        col1,col2=st.columns(2)
        states=sorted(list(master_zip['STATE_LONG'].astype(str).unique()))
        state_select=col1.multiselect('Filter Results by State(s)',states)
        count_select=col2.number_input(label='How many similar locations returned? (5-25)',min_value=5,max_value=25,value=10,step=5)
        st.subheader('Data Category Importance')
        st.caption('Lower values = lower importance, higher values = higher importnace, default = 1.0')
        #people_select=st.slider(label='People',min_value=0.1, max_value=2.0, step=0.1, value=1.0)
        home_select=st.slider(label='Home',min_value=0.1, max_value=2.0, step=0.1, value=1.0)
        #work_select=st.slider(label='Work',min_value=0.1, max_value=2.0, step=0.1, value=1.0)
        environment_select=st.slider(label='Environment',min_value=0.1, max_value=2.0, step=0.1, value=1.0)

    filt_master_zip=master_zip
    filt_master_city=master_city
    if len(state_select)>0:
        filt_master_zip=master_zip[master_zip['STATE_LONG'].isin(state_select)]
        filt_master_city=master_city[master_city['STATE_LONG'].isin(state_select)]

    #Benchmark
    if loc_select=='City':
        if city_select !='City':
            selected_record = master_city[master_city['CITYSTATE']==city_select].reset_index()
            selected_city=selected_record['CITYSTATE'][0]
            #selected_county=selected_record['County Title'][0]
            #Columns for scaling
            PeopleCols_sc=['MED_AGE_SC','PCT_UNDER_18_SC','MED_HH_INC_SC', 'PCT_POVERTY_SC','PCT_BACH_MORE_SC']
            HomeCols_sc=['HH_SIZE_SC','PCT_OWN_SC','MED_HOME_SC','PCT_UNIT1_SC','PCT_UNIT24_SC']
            #WorkCols_sc=['MEAN_COMMUTE_SC','PCT_WC_SC','PCT_WORKING_SC','PCT_SERVICE_SC','PCT_BC_SC']
            EnvironmentCols_sc=['PCT_WATER_SC','ENV_INDEX_SC','PCT_TOPARK_ONEMILE_SC','POP_DENSITY_SC','METRO_INDEX_SC']
            
            # Calculate the euclidian distance between the selected record and the rest of the dataset
            People_dist             = euclidean_distances(filt_master_city.loc[:, PeopleCols_sc], selected_record[PeopleCols_sc].values.reshape(1, -1))
            Home_dist               = euclidean_distances(filt_master_city.loc[:, HomeCols_sc], selected_record[HomeCols_sc].values.reshape(1, -1))
            #Work_dist               = euclidean_distances(filt_master_city.loc[:, WorkCols_sc], selected_record[WorkCols_sc].values.reshape(1, -1))
            Environment_dist        = euclidean_distances(filt_master_city.loc[:, EnvironmentCols_sc], selected_record[EnvironmentCols_sc].values.reshape(1, -1))

            # Create a new dataframe with the similarity scores and the corresponding index of each record
            df_similarity = pd.DataFrame({'HOME_SIM': Home_dist [:, 0], 'ENV_SIM': Environment_dist [:, 0], 'index': filt_master_city.index})

            #df_similarity['OVERALL_SIM']=df_similarity['PEOPLE_SIM','HOME_SIM','WORK_SIM','ENV_SIM'].mean(axis=1)
            weights=[home_select,environment_select]
            # Multiply column values with weights
            df_weighted = df_similarity.loc[:, ['HOME_SIM', 'ENV_SIM']].mul(weights)
            df_similarity['OVERALL_W']=df_weighted.sum(axis=1)/sum(weights)

            #people_max=df_similarity['PEOPLE_SIM'].max()
            home_max=df_similarity['HOME_SIM'].max()
            #work_max=df_similarity['WORK_SIM'].max()
            env_max=df_similarity['ENV_SIM'].max()
            overall_max=df_similarity['OVERALL_W'].max()

            #df_similarity['PEOPLE']  = 100 - (100 * df_similarity['PEOPLE_SIM'] / people_max)
            df_similarity['HOME']    = 100 - (100 * df_similarity['HOME_SIM'] / home_max)
            #df_similarity['WORK']    = 100 - (100 * df_similarity['WORK_SIM'] / work_max)
            df_similarity['ENVIRONMENT']     = 100 - (100 * df_similarity['ENV_SIM'] / env_max)
            df_similarity['OVERALL'] = 100 - (100 * df_similarity['OVERALL_W'] / overall_max)

            # Sort the dataframe by the similarity scores in descending order and select the top 10 most similar records
            df_similarity = df_similarity.sort_values(by='OVERALL_W', ascending=True).head(count_select+1)

            # Merge the original dataframe with the similarity dataframe to display the top 10 most similar records
            df_top10 = pd.merge(df_similarity, filt_master_city, left_on='index', right_index=True).reset_index(drop=True)
            df_top10=df_top10.loc[1:count_select]
            df_top10['Rank']=list(range(1,count_select+1))
            df_top10['Ranking']=df_top10['Rank'].astype(str)+'- '+df_top10['CITYSTATE']
            df_top10['LAT_R']=selected_record['LAT'][0]
            df_top10['LON_R']=selected_record['LON'][0]
            df_top10['SAVE']=False
            df_top10['NOTES']=''

            st.header('Top '+'{}'.format(count_select)+' Most Similar Locations')
            #st.write('You selected zip code '+zip_select+' from '+selected_record['County Title'][0])
            # CSS to inject contained in a string
            hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """

            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)
            tab1,tab2=st.tabs(['Map','Data'])
            with tab2:
                with st.expander('Expand for Table Info'):
                    st.markdown(
                    """
                    - The values for OVERALL, HOME, and ENVIRONMENT are scaled similarity scores for the respective categories with values of 0-100, where 100 represents a perfect match.
                    - Locations are ranked by their OVERALL score, which is a weighted average of the individual category scores.
                    - Save your research by checking locations in the SAVE column which will be added to csv for download.
                    """
                    )
                @st.cache_data
                def convert_df(df):
                    return df.to_csv().encode('utf-8')
                cols=['Rank','CITYSTATE','OVERALL','HOME','ENVIRONMENT']
                df=df_top10[cols+['SAVE','NOTES']]
                df=df.set_index('Rank')
                edited_df=st.experimental_data_editor(df)
                save=edited_df[edited_df['SAVE']==True]
                save=save.reset_index()
                csv = convert_df(save[cols+['SAVE','NOTES']])
                st.download_button(label="Download Selections as CSV",data=csv,file_name='SIMILO_SAVED.csv',mime='text/csv',)
            with tab1:
                latcenter=df_top10['LAT'].mean()
                loncenter=df_top10['LON'].mean()
                #map token for additional map layers
                token = "pk.eyJ1Ijoia3NvZGVyaG9sbTIyIiwiYSI6ImNsZjI2djJkOTBmazU0NHBqdzBvdjR2dzYifQ.9GkSN9FUYa86xldpQvCvxA" # you will need your own token
                #mapbox://styles/mapbox/streets-v12
                fig1 = px.scatter_mapbox(df_top10, lat='LAT',lon='LON',center=go.layout.mapbox.Center(lat=latcenter,lon=loncenter),
                                     color="Rank", color_continuous_scale=px.colors.sequential.ice, hover_name='CITYSTATE', hover_data=['Rank'],zoom=3,)
                fig1.update_traces(marker={'size': 15})
                fig1.update_layout(mapbox_style="mapbox://styles/mapbox/satellite-streets-v12",
                               mapbox_accesstoken=token)
                fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig1,use_container_width=True)

            st.divider()

            st.header('Location Deep Dive')
            rank_select=st.selectbox('Select from rankings above',list(df_top10['Ranking']))
            if rank_select:
                compare_record=df_top10[df_top10['Ranking']==rank_select].reset_index(drop=True)
                compare_city=compare_record['CITYSTATE'][0]
                #compare_county=compare_record['County Title'][0]
                compare_state=compare_record['STATE_SHORT'][0].lower()
                #st.write(selected_zip+' in '+selected_county+' VS '+compare_zip+' in '+compare_county)
                tab1,tab3,tab5 = st.tabs(['Overall','Home','Environment'])
                with tab1:
                    col1,col2=st.columns(2)
                    col1.subheader('Selected')
                    col1.write(selected_city)
                    col2.subheader('Similar')
                    col2.write(compare_city)
                    st.divider()
                    st.subheader('Similarity Scores')
                    col1,col2,col3=st.columns(3)
                    col1.metric('Overall',compare_record['OVERALL'][0].round(2))
                    col1.progress(compare_record['OVERALL'][0]/100)
                    #col2.metric('People',compare_record['PEOPLE'][0].round(2))
                    #col2.progress(compare_record['PEOPLE'][0]/100)
                    col2.metric('Home',compare_record['HOME'][0].round(2))
                    col2.progress(compare_record['HOME'][0]/100)
                    #col4.metric('Work',compare_record['WORK'][0].round(2))
                    #col4.progress(compare_record['WORK'][0]/100)
                    col3.metric('Environment',compare_record['ENVIRONMENT'][0].round(2))
                    col3.progress(compare_record['ENVIRONMENT'][0]/100)
                    df_long = pd.melt(compare_record[['OVERALL','HOME','ENVIRONMENT']].reset_index(), id_vars=['index'], var_name='Categories', value_name='Scores')
                    fig = px.bar(df_long, x='Categories', y='Scores', color='Scores', color_continuous_scale='blues')
                    fig.update_layout(xaxis_title='Categories',
                    yaxis_title='Similarity Scores')
                    st.plotly_chart(fig,use_container_width=True)    
                with tab3:
                    selected_record['PCT_18_65']=selected_record['PCT_OVER_18']-selected_record['PCT_OVER_65']
                    compare_record['PCT_18_65']=compare_record['PCT_OVER_18']-compare_record['PCT_OVER_65']
                    dif_cols=['MED_AGE','MED_HH_INC','PCT_POVERTY','PCT_BACH_MORE','POP_DENSITY','METRO_INDEX',
                        'HH_SIZE','FAM_SIZE','MED_HOME','MED_RENT','PCT_UNIT1','PCT_WORKING',
                        'MEAN_COMMUTE','PCT_WATER','ENV_INDEX','PCT_TOPARK_HALFMILE','PCT_TOPARK_ONEMILE']
                    dif_record=compare_record[dif_cols]-selected_record[dif_cols]
                    st.write(
                    """
                    <style>
                    [data-testid="stMetricDelta"] svg {
                    display: none;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                    )
                    
                    col1,col2=st.columns(2)
                    col1.subheader('Selected')
                    col1.write(selected_city)
                    col2.subheader('Similar')
                    col2.write(compare_city)
                    st.divider()
                    col1,col2=st.columns(2)
                    fig = px.pie(selected_record, values=[selected_record['PCT_OWN'][0], selected_record['PCT_RENT'][0]],names=['Percent Own Home','Percent Renting'])
                    fig.update_layout(legend={'title': {'text': 'Home Ownership'}})
                    col1.caption('Selected')
                    col1.plotly_chart(fig,use_container_width=True)
                    fig=px.pie(selected_record, values=[compare_record['PCT_OWN'][0], compare_record['PCT_RENT'][0]],names=['Percent Own Home','Percent Renting'])
                    fig.update_layout(legend={'title': {'text': 'Home Ownership'}})
                    col2.caption('Similar')
                    col2.plotly_chart(fig,use_container_width=True)
                    st.divider()
                    col1,col2=st.columns(2)
                    col1.caption('Selected')
                    col1.metric('Avg Household Size','{:,.1f}'.format(selected_record['HH_SIZE'][0].round(2)))
                    col2.caption('Similar')
                    col2.metric('Avg Household Size','{:,.1f}'.format(compare_record['HH_SIZE'][0].round(2)),delta='{:,.1f}'.format(dif_record['HH_SIZE'][0].round(2)))
                    st.divider()
                    col1,col2=st.columns(2) 
                    col1.caption('Selected')
                    col1.metric('Avg Family Size','{:,.1f}'.format(selected_record['FAM_SIZE'][0].round(2)))
                    col2.caption('Similar')
                    col2.metric('Avg Family Size','{:,.1f}'.format(compare_record['FAM_SIZE'][0].round(2)),delta='{:,.1f}'.format(dif_record['FAM_SIZE'][0].round(2)))
                    st.divider()
                    col1,col2=st.columns(2)
                    col1.caption('Selected')
                    col1.metric('Median Home Price','${:,.0f}'.format(selected_record['MED_HOME'][0].round(2)))
                    col2.caption('Similar')
                    col2.metric('Median Home Price','${:,.0f}'.format(compare_record['MED_HOME'][0].round(2)),delta='${:,.0f}'.format(dif_record['MED_HOME'][0].round(2)))
                    st.divider()
                    col1,col2=st.columns(2)
                    col1.caption('Selected')
                    col1.metric('Median Rent Price','${:,.0f}'.format(selected_record['MED_RENT'][0].round(2)))
                    col2.caption('Similar')
                    col2.metric('Median Rent Price','${:,.0f}'.format(compare_record['MED_RENT'][0].round(2)),delta='${:,.0f}'.format(dif_record['MED_RENT'][0].round(2)))
                    st.divider()
                    col1,col2=st.columns(2)
                    col1.caption('Selected')
                    col1.metric('% Single Family Residential','{:.1%}'.format(selected_record['PCT_UNIT1'][0].round(2)/100))
                    col2.caption('Similar')
                    col2.metric('% Single Family Residential','{:.1%}'.format(compare_record['PCT_UNIT1'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_UNIT1'][0].round(2)/100))
                with tab5:
                    col1,col2=st.columns(2)
                    col1.subheader('Selected')
                    col1.write(selected_city)
                    col2.subheader('Similar')
                    col2.write(compare_city)
                    st.divider()
                    col1,col2=st.columns(2)
                    col1.caption('Selected')
                    col1.write('Location Type')
                    col1.write(selected_record['METROPOLITAN'][0])
                    col2.caption('Similar')
                    col2.write('Location Type')
                    col2.write(compare_record['METROPOLITAN'][0])
                    st.divider()
                    col1,col2=st.columns(2)
                    col1.caption('Selected')
                    col1.metric('Population Density','{:,.0f}'.format(selected_record['POP_DENSITY'][0].round(2)))
                    col2.caption('Similar')
                    col2.metric('Population Density','{:,.0f}'.format(compare_record['POP_DENSITY'][0].round(2)),delta='{:.0f}'.format(dif_record['POP_DENSITY'][0]))
                    st.divider()
                    col1,col2=st.columns(2)
                    col1.caption('Selected')
                    col1.metric('% Area is Water','{:.2%}'.format(selected_record['PCT_WATER'][0]))
                    col2.caption('Similar')
                    col2.metric('% Area is Water','{:.2%}'.format(compare_record['PCT_WATER'][0]),delta='{:.2%}'.format(dif_record['PCT_WATER'][0]))
                    st.divider()
                    col1,col2=st.columns(2)
                    col1.caption('Selected')
                    col1.metric('Environmental Quality Index','{:.2f}'.format(selected_record['ENV_INDEX'][0].round(2)))
                    col2.caption('Similar')
                    col2.metric('Environmental Quality Index','{:.2f}'.format(compare_record['ENV_INDEX'][0].round(2)),delta='{:.2f}'.format(dif_record['ENV_INDEX'][0]))
                    #st.divider()
                    #col1,col2=st.columns(2)
                    #col1.caption('Selected')
                    #col1.metric('Pct within 0.5 mile to Park','{:.1%}'.format(selected_record['PCT_TOPARK_HALFMILE'][0].round(2)/100))
                    #col2.caption('Similar')
                    #col2.metric('Pct within 0.5 mile to Park','{:.1%}'.format(compare_record['PCT_TOPARK_HALFMILE'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_TOPARK_HALFMILE'][0]/100))
                    #st.divider()
                    #col1,col2=st.columns(2)
                    #col1.caption('Selected')
                    #col1.metric('Pct within 1 mile to Park','{:.1%}'.format(selected_record['PCT_TOPARK_ONEMILE'][0].round(2)/100))
                    #col2.caption('Similar')
                    #col2.metric('Pct within 1 mile to Park','{:.1%}'.format(compare_record['PCT_TOPARK_ONEMILE'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_TOPARK_ONEMILE'][0]/100))
                    
    
                   
    if zip_select != 'Zip':
        selected_record = master_zip[master_zip['ZIP']==zip_select].reset_index()
        selected_zip=selected_record['ZIP'][0]
        selected_county=selected_record['COUNTY_NAME'][0]
        selected_state=selected_record['STATE_SHORT'][0]

        #Columns for scaling
        #PeopleCols_sc=['MED_AGE_SC','PCT_UNDER_18_SC','MED_HH_INC_SC', 'PCT_POVERTY_SC','PCT_BACH_MORE_SC']
        HomeCols_sc=['HH_SIZE_SC','PCT_OWN_SC','MED_HOME_SC','PCT_UNIT1_SC','PCT_UNIT24_SC']
        #WorkCols_sc=['MEAN_COMMUTE_SC','PCT_WC_SC','PCT_WORKING_SC','PCT_SERVICE_SC','PCT_BC_SC']
        EnvironmentCols_sc=['PCT_WATER_SC','ENV_INDEX_SC','PCT_TOPARK_ONEMILE_SC','POP_DENSITY_SC','METRO_INDEX_SC']

        # Calculate the euclidian distance between the selected record and the rest of the dataset
        #People_dist             = euclidean_distances(filt_master_zip.loc[:, PeopleCols_sc], selected_record[PeopleCols_sc].values.reshape(1, -1))
        Home_dist               = euclidean_distances(filt_master_zip.loc[:, HomeCols_sc], selected_record[HomeCols_sc].values.reshape(1, -1))
        #Work_dist               = euclidean_distances(filt_master_zip.loc[:, WorkCols_sc], selected_record[WorkCols_sc].values.reshape(1, -1))
        Environment_dist        = euclidean_distances(filt_master_zip.loc[:, EnvironmentCols_sc], selected_record[EnvironmentCols_sc].values.reshape(1, -1))

        # Create a new dataframe with the similarity scores and the corresponding index of each record
        df_similarity = pd.DataFrame({'HOME_SIM': Home_dist [:, 0],'ENV_SIM': Environment_dist [:, 0], 'index': filt_master_zip.index})

        #df_similarity['OVERALL_SIM']=df_similarity['PEOPLE_SIM','HOME_SIM','WORK_SIM','ENV_SIM'].mean(axis=1)
        weights=[home_select,environment_select]
        # Multiply column values with weights
        df_weighted = df_similarity.loc[:, ['HOME_SIM', 'ENV_SIM']].mul(weights)
        df_similarity['OVERALL_W']=df_weighted.sum(axis=1)/sum(weights)

        #people_max=df_similarity['PEOPLE_SIM'].max()
        home_max=df_similarity['HOME_SIM'].max()
        #work_max=df_similarity['WORK_SIM'].max()
        env_max=df_similarity['ENV_SIM'].max()
        overall_max=df_similarity['OVERALL_W'].max()

        #df_similarity['PEOPLE']  = 100 - (100 * df_similarity['PEOPLE_SIM'] / people_max)
        df_similarity['HOME']    = 100 - (100 * df_similarity['HOME_SIM'] / home_max)
        #df_similarity['WORK']    = 100 - (100 * df_similarity['WORK_SIM'] / work_max)
        df_similarity['ENVIRONMENT']     = 100 - (100 * df_similarity['ENV_SIM'] / env_max)
        df_similarity['OVERALL'] = 100 - (100 * df_similarity['OVERALL_W'] / overall_max)

        # Sort the dataframe by the similarity scores in descending order and select the top 10 most similar records
        df_similarity = df_similarity.sort_values(by='OVERALL_W', ascending=True).head(count_select+1)

        # Merge the original dataframe with the similarity dataframe to display the top 10 most similar records
        df_top10 = pd.merge(df_similarity, filt_master_zip, left_on='index', right_index=True).reset_index(drop=True)
        df_top10=df_top10.loc[1:count_select]
        df_top10['RANK']=list(range(1,count_select+1))
        df_top10['RANKING']=df_top10['RANK'].astype(str)+' - Zip Code '+df_top10['ZIP']+' from '+df_top10['COUNTY_NAME']+' County, '+df_top10['STATE_SHORT']
        df_top10['LAT_R']=selected_record['LAT'][0]
        df_top10['LON_R']=selected_record['LON'][0]
        df_top10['SAVE']=False
        df_top10['NOTES']=''

        st.header('Top '+'{}'.format(count_select)+' Most Similar Locations')
        #st.write('You selected zip code '+zip_select+' from '+selected_record['County Title'][0])
        # CSS to inject contained in a string
        hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

        # Inject CSS with Markdown
        st.markdown(hide_table_row_index, unsafe_allow_html=True)
        tab1,tab2=st.tabs(['Map','Data'])
        with tab2:
            with st.expander('Expand for Table Info'):
                st.markdown(
                """
                - The values for OVERALL,HOME, and ENVIRONMENT are scaled similarity scores for the respective categories with values of 0-100, where 100 represents a perfect match.
                - Locations are ranked by their OVERALL score, which is a weighted average of the individual category scores.
                - Save your research by checking locations in the SAVE column which will be added to csv for download.
                """
                )
            @st.cache_data
            def convert_df(df):
                return df.to_csv().encode('utf-8')
            df_top10['COUNTY_STATE']=df_top10['COUNTY_NAME']+' County, '+df_top10['STATE_SHORT']
            cols=['ZIP','COUNTY_STATE','RANK','OVERALL','HOME','ENVIRONMENT']
            df=df_top10[cols+['SAVE','NOTES']]
            df=df.set_index('RANK')
            edited_df=st.experimental_data_editor(df)
            save=edited_df[edited_df['SAVE']==True]
            save=save.reset_index()
            csv = convert_df(save[cols+['SAVE','NOTES']])
            st.download_button(label="Download Selections as CSV",data=csv,file_name='SIMILO_SAVED.csv',mime='text/csv',)
        with tab1:
            latcenter=df_top10['LAT'].mean()
            loncenter=df_top10['LON'].mean()
            #map token for additional map layers
            token = "pk.eyJ1Ijoia3NvZGVyaG9sbTIyIiwiYSI6ImNsZjI2djJkOTBmazU0NHBqdzBvdjR2dzYifQ.9GkSN9FUYa86xldpQvCvxA" # you will need your own token
            #mapbox://styles/mapbox/streets-v12
            fig1 = px.scatter_mapbox(df_top10, lat='LAT',lon='LON',center=go.layout.mapbox.Center(lat=latcenter,lon=loncenter),
                                    color="RANK", color_continuous_scale=px.colors.sequential.ice, hover_name='ZIP', hover_data=['RANK','COUNTY_NAME'],zoom=3,)
            fig1.update_traces(marker={'size': 15})
            fig1.update_layout(mapbox_style="mapbox://styles/mapbox/satellite-streets-v12",
                               mapbox_accesstoken=token)
            fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig1,use_container_width=True)

        st.divider()

        st.header('Location Deep Dive')
        rank_select=st.selectbox('Select from rankings above',list(df_top10['RANKING']))
        if rank_select:
            compare_record=df_top10[df_top10['RANKING']==rank_select].reset_index(drop=True)
            compare_zip=compare_record['ZIP'][0]
            compare_county=compare_record['COUNTY_NAME'][0]
            compare_state=compare_record['STATE_SHORT'][0]
            #st.write(selected_zip+' in '+selected_county+' VS '+compare_zip+' in '+compare_county)
            tab1,tab3,tab5 = st.tabs(['Overall','Home','Environment'])
            with tab1:
                col1,col2=st.columns(2)
                col1.subheader('Selected')
                col1.write(selected_zip+' in '+selected_county+' County, '+selected_state)
                col2.subheader('Similar')
                col2.write(compare_zip+' in '+compare_county+' County, '+compare_state)
                st.divider()
                st.subheader('Similarity Scores')
                col1,col2,col3=st.columns(3)
                col1.metric('Overall',compare_record['OVERALL'][0].round(2))
                col1.progress(compare_record['OVERALL'][0]/100)
                #col2.metric('People',compare_record['PEOPLE'][0].round(2))
                #col2.progress(compare_record['PEOPLE'][0]/100)
                col2.metric('Home',compare_record['HOME'][0].round(2))
                col2.progress(compare_record['HOME'][0]/100)
                #col4.metric('Work',compare_record['WORK'][0].round(2))
                #col4.progress(compare_record['WORK'][0]/100)
                col3.metric('Environment',compare_record['ENVIRONMENT'][0].round(2))
                col3.progress(compare_record['ENVIRONMENT'][0]/100)
                df_long = pd.melt(compare_record[['OVERALL','HOME','ENVIRONMENT']].reset_index(), id_vars=['index'], var_name='Categories', value_name='Scores')
                fig = px.bar(df_long, x='Categories', y='Scores', color='Scores', color_continuous_scale='blues')
                fig.update_layout(xaxis_title='Categories',
                  yaxis_title='Similarity Scores')
                st.plotly_chart(fig,use_container_width=True)   
            with tab3:
                selected_record['PCT_18_65']=selected_record['PCT_OVER_18']-selected_record['PCT_OVER_65']
                compare_record['PCT_18_65']=compare_record['PCT_OVER_18']-compare_record['PCT_OVER_65']
                dif_cols=['MED_AGE','MED_HH_INC','PCT_POVERTY','PCT_BACH_MORE','POP_DENSITY','METRO_INDEX',
                        'HH_SIZE','FAM_SIZE','MED_HOME','MED_RENT','PCT_UNIT1','PCT_WORKING',
                        'MEAN_COMMUTE','PCT_WATER','ENV_INDEX','PCT_TOPARK_HALFMILE','PCT_TOPARK_ONEMILE']
                dif_record=compare_record[dif_cols]-selected_record[dif_cols]
                st.write(
                """
                <style>
                [data-testid="stMetricDelta"] svg {
                display: none;
                }
                </style>
                """,
                unsafe_allow_html=True,
                )

                col1,col2=st.columns(2)
                col1.subheader('Selected')
                col1.write(selected_zip+' in '+selected_county+' County, '+selected_state)
                col2.subheader('Similar')
                col2.write(compare_zip+' in '+compare_county+' County, '+compare_state)
                st.divider()
                col1,col2=st.columns(2)
                fig = px.pie(selected_record, values=[selected_record['PCT_OWN'][0], selected_record['PCT_RENT'][0]],names=['Percent Own Home','Percent Renting'])
                fig.update_layout(legend={'title': {'text': 'Home Ownership'}})
                col1.caption('Selected')
                col1.plotly_chart(fig,use_container_width=True)
                fig=px.pie(selected_record, values=[compare_record['PCT_OWN'][0], compare_record['PCT_RENT'][0]],names=['Percent Own Home','Percent Renting'])
                fig.update_layout(legend={'title': {'text': 'Home Ownership'}})
                col2.caption('Similar')
                col2.plotly_chart(fig,use_container_width=True)
                st.divider()
                col1,col2=st.columns(2)
                col1.caption('Selected')
                col1.metric('Avg Household Size','{:,.1f}'.format(selected_record['HH_SIZE'][0].round(2)))
                col2.caption('Similar')
                col2.metric('Avg Household Size','{:,.1f}'.format(compare_record['HH_SIZE'][0].round(2)),delta='{:,.1f}'.format(dif_record['HH_SIZE'][0].round(2)))
                st.divider()
                col1,col2=st.columns(2) 
                col1.caption('Selected')
                col1.metric('Avg Family Size','{:,.1f}'.format(selected_record['FAM_SIZE'][0].round(2)))
                col2.caption('Similar')
                col2.metric('Avg Family Size','{:,.1f}'.format(compare_record['FAM_SIZE'][0].round(2)),delta='{:,.1f}'.format(dif_record['FAM_SIZE'][0].round(2)))
                st.divider()
                col1,col2=st.columns(2)
                col1.caption('Selected')
                col1.metric('Median Home Price','${:,.0f}'.format(selected_record['MED_HOME'][0].round(2)))
                col2.caption('Similar')
                col2.metric('Median Home Price','${:,.0f}'.format(compare_record['MED_HOME'][0].round(2)),delta='${:,.0f}'.format(dif_record['MED_HOME'][0].round(2)))
                st.divider()
                col1,col2=st.columns(2)
                col1.caption('Selected')
                col1.metric('Median Rent Price','${:,.0f}'.format(selected_record['MED_RENT'][0].round(2)))
                col2.caption('Similar')
                col2.metric('Median Rent Price','${:,.0f}'.format(compare_record['MED_RENT'][0].round(2)),delta='${:,.0f}'.format(dif_record['MED_RENT'][0].round(2)))
                st.divider()
                col1,col2=st.columns(2)
                col1.caption('Selected')
                col1.metric('% Single Family Residential','{:.1%}'.format(selected_record['PCT_UNIT1'][0].round(2)/100))
                col2.caption('Similar')
                col2.metric('% Single Family Residential','{:.1%}'.format(compare_record['PCT_UNIT1'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_UNIT1'][0].round(2)/100))
            with tab5:
                col1,col2=st.columns(2)
                col1.subheader('Selected')
                col1.write(selected_zip+' in '+selected_county+' County, '+selected_state)
                col2.subheader('Similar')
                col2.write(compare_zip+' in '+compare_county+' County, '+compare_state)
                st.divider()
                col1,col2=st.columns(2)
                col1.caption('Selected')
                col1.write('Location Type')
                col1.write(selected_record['METROPOLITAN'][0])
                col2.caption('Similar')
                col2.write('Location Type')
                col2.write(compare_record['METROPOLITAN'][0])
                st.divider()
                col1,col2=st.columns(2)
                col1.caption('Selected')
                col1.metric('Population Density','{:,.0f}'.format(selected_record['POP_DENSITY'][0].round(2)))
                col2.caption('Similar')
                col2.metric('Population Density','{:,.0f}'.format(compare_record['POP_DENSITY'][0].round(2)),delta='{:.0f}'.format(dif_record['POP_DENSITY'][0]))
                st.divider()
                col1,col2=st.columns(2)
                col1.caption('Selected')
                col1.metric('% Area is Water','{:.2%}'.format(selected_record['PCT_WATER'][0]))
                col2.caption('Similar')
                col2.metric('% Area is Water','{:.2%}'.format(compare_record['PCT_WATER'][0]),delta='{:.2%}'.format(dif_record['PCT_WATER'][0]))
                st.divider()
                col1,col2=st.columns(2)
                col1.caption('Selected')
                col1.metric('Environmental Quality Index','{:.2f}'.format(selected_record['ENV_INDEX'][0].round(2)))
                col2.caption('Similar')
                col2.metric('Environmental Quality Index','{:.2f}'.format(compare_record['ENV_INDEX'][0].round(2)),delta='{:.2f}'.format(dif_record['ENV_INDEX'][0]))
                #st.divider()
                #col1,col2=st.columns(2)
                #col1.caption('Selected')
                #col1.metric('Pct within 0.5 mile to Park','{:.1%}'.format(selected_record['PCT_TOPARK_HALFMILE'][0].round(2)/100))
                #col2.caption('Similar')
                #col2.metric('Pct within 0.5 mile to Park','{:.1%}'.format(compare_record['PCT_TOPARK_HALFMILE'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_TOPARK_HALFMILE'][0]/100))
                #st.divider()
                #col1,col2=st.columns(2)
                #col1.caption('Selected')
                #col1.metric('Pct within 1 mile to Park','{:.1%}'.format(selected_record['PCT_TOPARK_ONEMILE'][0].round(2)/100))
                #col2.caption('Similar')
                #col2.metric('Pct within 1 mile to Park','{:.1%}'.format(compare_record['PCT_TOPARK_ONEMILE'][0].round(2)/100),delta='{:.1%}'.format(dif_record['PCT_TOPARK_ONEMILE'][0]/100))
                                 

#About Page
if selected=='Analyze':
    #st.title('Data')
    #st.subheader('All data for this project was publicly sourced from:')
    # Setup for Storytelling (matplotlib):
    plt.rcParams['font.family'] = 'monospace'
    plt.rcParams['font.size'] = 8
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['figure.facecolor'] = '#464545' 
    plt.rcParams['axes.facecolor'] = '#464545' 
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.titlesize'] = 9
    plt.rcParams['axes.labelcolor'] = 'darkgray'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.edgecolor'] = 'darkgray'
    plt.rcParams['axes.linewidth'] = 0.2
    plt.rcParams['ytick.color'] = 'darkgray'
    plt.rcParams['xtick.color'] = 'darkgray'
    plt.rcParams['axes.titlecolor'] = '#FFFFFF'
    plt.rcParams['axes.titlecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'darkgray'
    plt.rcParams['axes.linewidth'] = 0.85
    plt.rcParams['ytick.major.size'] = 0

    st.sidebar.markdown(''' > **How to analyze**

    1. To Select a state (**green dot**).
    2. To compare for the selected state against other 50 states (**white dots**).
    3. To compare the chosen state against **national average** and the data distribution.
    4. To extract insights as "An appreciation above national average & price below average = possible *opportunity*".
    ''')

    # --- App (begin):
    US_real_estate_appreciation = pd.read_csv('data/appreciation_Q2_2023.csv')
    #US_real_estate_appreciation['Annual_appreciation'] = round(US_real_estate_appreciation['Annual_appreciation'], 2)*100

    # Page setup:
    #st.set_page_config(
    #    page_title="Residential properties United States",
    #    page_icon="🏢",
    #    layout="centered",
    #    initial_sidebar_state="expanded",
    #)

    # Header:
    st.header('Appreciation of residential properties in United States')

    


    # Widgets:
    states = sorted(list(US_real_estate_appreciation['Location'].unique()))
    state_selection = st.selectbox(
        '🌎 Select a state',
        states
    )

    # State selection:
    your_state = state_selection
    selected_state = US_real_estate_appreciation.query('Location == @your_state')
    other_states = US_real_estate_appreciation.query('Location != @your_state')

    # CHART 1: Annual appreciation (Q2 2023):
    chart_1, ax = plt.subplots(figsize=(3, 4.125))
    # Background:
    sns.stripplot(
        data= other_states,
        y = 'Annual_appreciation',
        color = 'white',
        jitter=0.85,
        size=8,
        linewidth=1,
        edgecolor='gainsboro',
        alpha=0.7
    )
    # Highlight:
    sns.stripplot(
        data= selected_state,
        y = 'Annual_appreciation',
        color = '#00FF7F',
        jitter=0.15,
        size=12,
        linewidth=1,
        edgecolor='k',
        label=f'{your_state}'
    )

    # Showing up position measures:
    avg_annual_val = US_real_estate_appreciation['Annual_appreciation'].median()
    q1_annual_val = np.percentile(US_real_estate_appreciation['Annual_appreciation'], 25)
    q3_annual_val = np.percentile(US_real_estate_appreciation['Annual_appreciation'], 75)

    # Plotting lines (reference):
    ax.axhline(y=avg_annual_val, color='#DA70D6', linestyle='--', lw=0.75)
    ax.axhline(y=q1_annual_val, color='white', linestyle='--', lw=0.75)
    ax.axhline(y=q3_annual_val, color='white', linestyle='--', lw=0.75)

    # Adding the labels for position measures:
    ax.text(1.15, q1_annual_val, 'Q1', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.text(1.3, avg_annual_val, 'Median', ha='center', va='center', color='#DA70D6', fontsize=8, fontweight='bold')
    ax.text(1.15, q3_annual_val, 'Q3', ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    # to fill the area between the lines:
    ax.fill_betweenx([q1_annual_val, q3_annual_val], -2, 1, alpha=0.2, color='gray')
    # to set the x-axis limits to show the full range of the data:
    ax.set_xlim(-1, 1)

    # Axes and titles:
    plt.xticks([])
    plt.ylabel('Average appreciation (%)')
    plt.title('Appreciation (%) as of Q2 2023', weight='bold', loc='center', pad=15, color='gainsboro')
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=0, labelcolor='#00FF7F')
    plt.tight_layout()


    # CHART 2: Price in 100K($):
    chart_2, ax = plt.subplots(figsize=(3, 3.95))
    # Background:
    sns.stripplot(
        data= other_states,
        y = 'Median_house_price',
        color = 'white',
        jitter=0.95,
        size=8,
        linewidth=1,
        edgecolor='gainsboro',
        alpha=0.7
    )
    # Highlight:
    sns.stripplot(
        data= selected_state,
        y = 'Median_house_price',
        color = '#00FF7F',
        jitter=0.15,
        size=12,
        linewidth=1,
        edgecolor='k',
        label=f'{your_state}'
    )

    # Showing up position measures:
    avg_price_m2 = US_real_estate_appreciation['Median_house_price'].median()
    q1_price_m2 = np.percentile(US_real_estate_appreciation['Median_house_price'], 25)
    q3_price_m2 = np.percentile(US_real_estate_appreciation['Median_house_price'], 75)

    # Plotting lines (reference):
    ax.axhline(y=avg_price_m2, color='#DA70D6', linestyle='--', lw=0.75)
    ax.axhline(y=q1_price_m2, color='white', linestyle='--', lw=0.75)
    ax.axhline(y=q3_price_m2, color='white', linestyle='--', lw=0.75)

    # Adding the labels for position measures:
    ax.text(1.15, q1_price_m2, 'Q1', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    ax.text(1.35, avg_price_m2, 'Median', ha='center', va='center', color='#DA70D6', fontsize=8, fontweight='bold')
    ax.text(1.15, q3_price_m2, 'Q3', ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    # to fill the area between the lines:
    ax.fill_betweenx([q1_price_m2, q3_price_m2], -2, 1, alpha=0.2, color='gray')
    # to set the x-axis limits to show the full range of the data:
    ax.set_xlim(-1, 1)

    # Axes and titles:
    plt.xticks([])
    plt.ylabel('Price in 100K($)')
    plt.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=0, labelcolor='#00FF7F')
    plt.title('Median house prices ($)', weight='bold', loc='center', pad=15, color='gainsboro')
    plt.tight_layout()

    # Splitting the charts into two columns:
    left, right = st.columns(2)

    # Columns (content):
    with left:
        st.pyplot(chart_1)
    with right:
        st.pyplot(chart_2)

    # Informational text:
    st.markdown('''
    <span style="color:white;font-size:10pt"> ⚪ Each point represents a state </span>
    <span style="color:#DA70D6;font-size:10pt"> ▫ <b> Average value </b></span>
    <span style="color:white;font-size:10pt"> ◽ Lowest values (<b> bottom </b>)
    ◽ Highest values (<b> top </b>) <br>
    ◽ **Q1** (first quartile): where 25% of data falls under
    ◽ **Q3** (third quartile): where 75% of data falls under
    </span>

    ''',unsafe_allow_html=True)

    # Showing up the numerical data (as a dataframe):
    st.dataframe(
        US_real_estate_appreciation.query('Location == @your_state')[[
        'Location', 'Annual_appreciation', 
        'Median_house_price']]
    )

    # Adding some reference indexes:
    st.markdown(''' > **Reference indexes (inflation):**

    * IPCA: **3.1%** (National Broad Consumer Price Index)
    * IGP-M: **2.9%** (General Market Price Index)

    > Data based on public informs that accounts residential properties for 50 states of United States(second quarter of 2023).
    ''')

    # --- (End of the App)

