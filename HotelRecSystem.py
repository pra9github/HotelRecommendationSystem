import streamlit as st
import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')

# Load data
hotel_details = pd.read_csv("hotels dataset/Hotel_details.csv", delimiter=',')
hotel_rooms = pd.read_csv("hotels dataset/Hotel_Room_attributes.csv", delimiter=',')
hotel_cost = pd.read_csv("hotels dataset/hotels_RoomPrice.csv", delimiter=',')

# Remove unnecessary columns
del hotel_details['id']
del hotel_rooms['id']
del hotel_details['zipcode']
hotel_details = hotel_details.dropna()
hotel_rooms = hotel_rooms.dropna()
hotel_details.drop_duplicates(subset='hotelid', keep=False, inplace=True)

# Merge dataframes
hotel = pd.merge(hotel_rooms, hotel_details, left_on='hotelcode', right_on='hotelid', how='inner')
del hotel['hotelid']
del hotel['url']
del hotel['curr']
del hotel['Source']

# Function to calculate number of guests
def calc():
    room_no = [('king', 2), ('queen', 2), ('triple', 3), ('master', 3), ('family', 4), ('murphy', 2), 
               ('quad', 4), ('double-double', 4), ('mini', 2), ('studio', 1), ('junior', 2), 
               ('apartment', 4), ('double', 2), ('twin', 2), ('double-twin', 4), ('single', 1), 
               ('diabled', 1), ('accessible', 1), ('suite', 2), ('one', 2)]
    guests_no = []
    for i in range(hotel.shape[0]):
        temp = hotel['roomtype'][i].lower().split()
        flag = 0
        for j in range(len(temp)):
            for k in range(len(room_no)):
                if temp[j] == room_no[k][0]:
                    guests_no.append(room_no[k][1])
                    flag = 1
                    break
            if flag == 1:
                break
        if flag == 0:
            guests_no.append(2)
    hotel['guests_no'] = guests_no

# Call calc() function to calculate guests_no
calc()

# Function to filter hotels by city
def city_based(city):
    city = city.lower()
    city_base = hotel[hotel['city'].str.lower() == city]
    city_base = city_base.sort_values(by='starrating', ascending=False)
    city_base.drop_duplicates(subset='hotelcode', keep='first', inplace=True)
    if not city_base.empty:
        hname = city_base[['hotelname', 'starrating', 'address', 'roomamenities', 'ratedescription']]
        return hname.head()
    else:
        st.write('No Hotels Available')

# Function to filter hotels by requirements
def requirement_based(city, number, features):
    city = city.lower()
    hotel['city'] = hotel['city'].str.lower()
    hotel['roomamenities'] = hotel['roomamenities'].str.lower()
    
    # Tokenize and lemmatize the features
    features_tokens = word_tokenize(features)  
    sw = stopwords.words('english')
    lemm = WordNetLemmatizer()
    f_set = {lemm.lemmatize(w) for w in features_tokens if w not in sw}
    
    # Filter hotels by city and number of guests
    req_based = hotel[(hotel['city'] == city) & (hotel['guests_no'] == number)]
    
    # Calculate similarity based on room amenities
    cos = []
    for amenities in req_based['roomamenities']:
        temp_tokens = word_tokenize(amenities)
        temp_set = {lemm.lemmatize(w) for w in temp_tokens if w not in sw}
        similarity = len(f_set.intersection(temp_set))
        cos.append(similarity)
    
    req_based['similarity'] = cos
    
    # Sort by similarity and return top recommendations
    req_based = req_based.sort_values(by='similarity', ascending=False)
    req_based = req_based.drop_duplicates(subset='hotelcode', keep='first').head(10)
    
    return req_based[['city', 'hotelname', 'roomtype', 'guests_no', 'starrating', 'address', 'roomamenities', 'ratedescription', 'similarity']]

# Streamlit UI
st.title('Hotel Recommendation System')

# Sidebar
# option = st.sidebar.selectbox('Select Option:', ('City Based', 'Requirement Based', 'Rate Based'))
option = st.sidebar.selectbox('Select Option:', ('Requirement Based',))

if option == 'City Based':
    city = st.sidebar.text_input('Enter City:', 'London')
    st.write('Top 5 hotels in', city.title())
    city_based(city)

elif option == 'Requirement Based':
    city = st.sidebar.text_input('Enter City:', 'London')
    number = st.sidebar.number_input('Number of Guests:', min_value=1, value=4)
    features = st.sidebar.text_input('Enter Features:', 'I need air conditioned room. I should have an alarm clock.')
    st.write('Recommendations based on your requirements:')
    if city:
        recommendations = requirement_based(city, number, features)
        st.write(recommendations)
    else:
        st.write('Please enter a city to get recommendations.')
