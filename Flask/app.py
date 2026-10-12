from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path


#Instantiate Flask application.
app = Flask(__name__)


#Define the location of the application files.
BASE_DIR = Path(__file__).resolve().parent


#Load the saved preprocessing and Gradient Boosting pipeline.
model = joblib.load(
    BASE_DIR / 'diabetes_risk_model.joblib'
)


#Define poverty thresholds by household size.
poverty_thresholds = {
    1: 16000,
    2: 21640,
    3: 27320,
    4: 33000,
    5: 38680,
    6: 44360
}


@app.route('/')
def home():

    return render_template(
        'index.html'
    )


@app.route('/predict', methods=['POST'])
def predict():

    #Retrieve demographic information.
    age = int(
        request.form['age']
    )

    sex = request.form['sex']

    race = request.form['race']


    #Retrieve household income information.
    household_income = float(
        request.form['household_income']
    )

    household_size = int(
        request.form['household_size']
    )


    #Calculate income-to-poverty ratio.
    income_ratio = (
        household_income /
        poverty_thresholds[household_size]
    )


    #Cap income ratio at 5 to match the training data.
    income_ratio = min(
        income_ratio,
        5.0
    )


    #Retrieve height and weight.
    height = float(
        request.form['height']
    )

    weight = float(
        request.form['weight']
    )


    #Calculate BMI using pounds and inches.
    bmi = (
        weight * 703
    ) / (
        height ** 2
    )


    #Retrieve behavioral variables.
    smoked_100 = int(
        request.form['smoked_100']
    )

    drank_heavily = int(
        request.form['drank_heavily']
    )

    weekly_exercise = int(
        request.form['weekly_exercise']
    )


    #Retrieve sedentary time in hours.
    sedentary_hours = float(
        request.form['sedentary_hours']
    )


    #Convert sedentary time to minutes
    #because the model expects minutes per day.
    sedentary_mins = (
        sedentary_hours * 60
    )


    #Create dataframe matching the model's input features.
    person = pd.DataFrame({

        'age': [age],

        'sex': [sex],

        'race': [race],

        'income_ratio': [
            income_ratio
        ],

        'bmi': [
            bmi
        ],

        'smoked_100': [
            smoked_100
        ],

        'drank_heavily': [
            drank_heavily
        ],

        'weekly_exercise': [
            weekly_exercise
        ],

        'sedentary_mins': [
            sedentary_mins
        ]

    })


    #Generate predicted diabetes probability.
    probability = (
        model
        .predict_proba(person)[0, 1]
        * 100
    )


    #Convert height into feet and inches for display.
    total_inches = int(height)

    height_feet = (
        total_inches // 12
    )

    height_inches = (
        total_inches % 12
    )

    height_display = (
        f"{height_feet}'{height_inches}\""
    )


    #Display prediction result.
    return render_template(

        'result.html',

        probability=round(
            probability,
            1
        ),

        age=age,

        sex=sex,

        race=race,

        household_income=int(
            household_income
        ),

        household_size=
            household_size,

        income_ratio=round(
            income_ratio,
            2
        ),

        height=
            height_display,

        weight=int(
            weight
        ),

        bmi=round(
            bmi,
            1
        ),

        smoked_100=
            smoked_100,

        drank_heavily=
            drank_heavily,

        weekly_exercise=
            weekly_exercise,

        sedentary_hours=
            sedentary_hours,

        sedentary_mins=int(
            sedentary_mins
        )

    )


@app.route('/about')
def about():

    return render_template(
        'about.html'
    )


if __name__ == '__main__':

    app.run(
        debug=True
    )