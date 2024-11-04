import streamlit as st

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
    # Displaying a fake travel plan
    st.markdown("""
# Hokkaido 5-Day Itinerary (December)

## Day 1: Arrival in Sapporo
- **Morning:**
  - Arrive at New Chitose Airport, Sapporo.
  - Take the JR Rapid Airport train to Sapporo Station (approx. 40 minutes).
  - Check-in at a family-friendly hotel in Sapporo, such as **JR Tower Hotel Nikko Sapporo**.

- **Afternoon:**
  - Visit the **Sapporo TV Tower** for panoramic views of the city.
  - Explore **Odori Park**, which is often beautifully decorated with winter lights.
  - Have lunch at a nearby café with kid-friendly options.

- **Evening:**
  - Dinner at **Ramen Alley** in Susukino for authentic Sapporo ramen.
  - Return to the hotel for a restful night.

## Day 2: Sapporo City Exploration
- **Morning:**
  - Breakfast at the hotel.
  - Visit **Shiroi Koibito Park** for a fun, chocolate-themed experience. Participate in cookie-making workshops, enjoyable for parents and a treat for the kid.

- **Afternoon:**
  - Lunch at the park's café.
  - Head to the **Sapporo Science Center** which offers kid-friendly exhibits and a planetarium.

- **Evening:**
  - Dine at **Sapporo Beer Garden**, where parents can enjoy local beer while selections for children are available.
  - Return to the hotel.

## Day 3: Day Trip to Otaru
- **Morning:**
  - Breakfast at the hotel.
  - Take a train from Sapporo Station to **Otaru** (approx. 30 minutes).
  - Visit the **Otaru Canal** for a scenic winter walk.

- **Afternoon:**
  - Enjoy lunch at one of the canal-side restaurants featuring fresh seafood.
  - Explore the **Otaru Music Box Museum** and create your own music box, a memorable activity for the family.
  - Visit the **Otaru Aquarium**.

- **Evening:**
  - Return to Sapporo by train.
  - Have dinner at the hotel or a nearby family-friendly restaurant.

## Day 4: Day Trip to Asahikawa
- **Morning:**
  - Early breakfast at the hotel.
  - Take a train to **Asahikawa** (approx. 1.5 hours).
  - Visit **Asahiyama Zoo**, famous for its interactive animal exhibits.

- **Afternoon:**
  - Lunch at the zoo or nearby restaurant.
  - Explore the **Snow Crystal Museum**, which looks like a small castle – enjoyable for kids with its enchanting architecture.

- **Evening:**
  - Head back to Sapporo by train.
  - Dinner at **Sapporo Station Shopping Plaza**, offering various cuisines.

## Day 5: Sapporo to Niseko and Departure
- **Morning:**
  - Check out from the hotel after breakfast.
  - Take a private car or shuttle to **Niseko** (approx. 2-3 hours, depending on weather conditions).
  - Enjoy a morning of snow activities tailored for families, such as snowman building or a scenic gondola ride.

- **Afternoon:**
  - Relax and have lunch at a ski resort lodge.
  - Explore Niseko town and shop for souvenirs.

- **Evening:**
  - Return to Sapporo or go directly to New Chitose Airport in the evening for departure.

---

### Tips:
- Always check weather forecasts and travel advisories as December can be snowy.
- Book accommodations and major travel tickets in advance given it's peak season.
- Many attractions provide rental services for winter gear, but be sure to bring warm clothes, especially for the toddler.

Enjoy your winter adventure in Hokkaido with your family!
""")