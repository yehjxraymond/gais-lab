import streamlit as st
from config import llm_open_router

# Title
st.title("Custom Travel Plan Generator")

# User Inputs
location = st.text_input("Enter your ideal travel location")
dates = st.date_input("Select the dates you'll be there", [])
number_of_people = st.number_input("Enter the number of people traveling", min_value=1, format="%d")
activity_focus = st.multiselect("Select the type of activities you're interested in", ["Cultural", "Adventure", "Relaxation", "Nature", "Food"])
pace = st.selectbox("Select your preferred pace of the trip", ["Relaxed", "Moderate", "Packed"])

# Generate Travel Plan Button
if st.button("Generate Travel Plan"):
    # Constructing the prompt with user inputs
    prompt = f"""
    Generate a detailed travel plan for the following details:
    Location: {location}
    Dates: {dates}
    Number of people: {number_of_people}
    Activity focus: {', '.join(activity_focus)}
    Pace of the trip: {pace}
    The plan should include specific activities, best places to visit, recommended food to try, and any other relevant advice. Format the plan in markdown for clarity.
    """
    
    # Using the llm_open_router to generate the travel plan
    response = llm_open_router.invoke(prompt)
    
    # Displaying the generated travel plan
    st.markdown(response.content)