from flask import Flask, render_template, request
from flask_mail import Mail, Message
import joblib
import pandas as pd
import urllib.parse

# Initialize Flask app
app = Flask(__name__)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'storyteller.owner46@gmail.com'         # your email
app.config['MAIL_PASSWORD'] = 'ayjo fryb tuqf lgjp'            # app password (not your normal password)
app.config['MAIL_DEFAULT_SENDER'] = 'storyteller.owner46@gmail.com'

mail = Mail(app)

# Load trained model and encoders
et_model = joblib.load("model/extra_trees_model.pkl")  
label_encoders = joblib.load("model/label_encoders.pkl")

# Load restaurant dataset for recommendations
restaurant_data = pd.read_csv("data/cleaned_restaurant_data.csv")

# Define feature names
feature_names = ['online_order', 'book_table', 'votes', 'location', 'rest_type', 'cuisines', 'cost']

# Extract unique locations from dataset
locations = sorted(restaurant_data["location"].dropna().unique())
cuisines = sorted(restaurant_data["cuisines"].dropna().unique())
rest_types = sorted(restaurant_data["rest_type"].dropna().unique())


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Email sending logic here
        subject = f"New Feedback from {name}"
        body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        msg = Message(subject=subject, recipients=['feedbacktracer@gmail.com'], body=body)
        mail.send(msg)

        # After submission, display thank you message
        return render_template('feedback.html', submitted=True, name=name)

    # If it's a GET request, show the feedback form
    return render_template('feedback.html', submitted=False)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def index():
    return render_template('index.html', 
                           locations=locations, 
                           cuisines=cuisines, 
                           rest_types=rest_types)  # Pass locations to template

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    user_online_order = int(request.form['online_order'])
    user_book_table = int(request.form['book_table'])
    user_votes = int(request.form['votes'])
    user_cost = int(request.form['cost'])
    user_location = request.form['location']
    user_cuisine = request.form['cuisine']
    user_rest_type = request.form['rest_type']

    # Encode categorical values
    location_encoded = label_encoders["location"].transform([user_location])[0]
    cuisine_encoded = label_encoders["cuisines"].transform([user_cuisine])[0]
    rest_type_encoded = label_encoders["rest_type"].transform([user_rest_type])[0]
    
    # Create DataFrame with correct feature order
    user_features_df = pd.DataFrame([[user_online_order, user_book_table, user_votes, 
                                      location_encoded, rest_type_encoded, cuisine_encoded, user_cost]], 
                                    columns=feature_names)

    # Make prediction
    predicted_rating = et_model.predict(user_features_df)[0]
    predicted_rating = round(predicted_rating, 2)  # Round to 2 decimal places
    
    # Find restaurants in user's location
    filtered_restaurants = restaurant_data[restaurant_data["location"] == user_location]

    if not filtered_restaurants.empty:
        # Find restaurants with rating close to predicted rating (±0.2 range)
        close_match = filtered_restaurants[
            (filtered_restaurants["rating"] >= predicted_rating - 0.2) & 
            (filtered_restaurants["rating"] <= predicted_rating + 0.2)
        ]

        # If we found matching restaurants, pick the one with highest votes
        if not close_match.empty:
            best_restaurant = close_match.nlargest(1, 'votes')
        else:
            # If no exact match, pick the highest-rated restaurant in the location
            best_restaurant = filtered_restaurants.nlargest(1, 'rating')

        restaurant_name = best_restaurant["name"].values[0]  
        restaurant_location = best_restaurant["location"].values[0]

        recommended_restaurant = best_restaurant["name"].values[0]
        famous_dish = best_restaurant["dish_liked"].values[0]
        google_maps_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote(restaurant_name + ', ' + restaurant_location)}"
    else:
        recommended_restaurant = "No matching restaurant found"
        famous_dish = "N/A"
        google_maps_link = "#"

    # Display a message if no restaurants were found
    return render_template('result.html', rating=predicted_rating, 
                           restaurant=recommended_restaurant, 
                           dish=famous_dish, location=user_location,
                           google_maps_link=google_maps_link)

if __name__ == '__main__':
    app.run(debug=True)
