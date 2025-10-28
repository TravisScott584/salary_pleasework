
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
import os
import requests
# Load model + encoders


MODEL_URL = "https://huggingface.co/TravisScott584/salary-predictor-model/resolve/main/salary_model_r_match.pkl"
MODEL_PATH = "salary_model_r_match.pkl"

if not os.path.exists(MODEL_PATH):
    st.write("Downloading model from Hugging Face...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.write("Model downloaded successfully.")

# Load model
saved = joblib.load(MODEL_PATH)
model = saved["model"]
encoders = saved["encoders"]
feature_order = saved["feature_order"]











# Application

st.title("Salary Prediction")
st.text("The model and application were created for the purposes of the AI101 final project. The goal of this software is to better inform students about their prospective careers.")
st.subheader("Fill out these fields as if you are getting the job now")
st.text("Ex: even if you are 18 now, enter the age you will be when you expect to secure this job.")

age = st.number_input("Enter your age when you expect to get this job (19-35):",
    min_value=19, max_value=35)
sex = st.selectbox("Gender:", ["M", "F"])
race = st.selectbox("Race/Ethnicity:", 
                    ["Asian", "American Indian", "Native Alaskan",
                     "Black", "Hispanic (any race)", "White",
                     "Pacific Islander", "Multi-Racial"])
degree_type = st.selectbox("Highest degree type:",
                           ["Bachelor's",
                            "Master's",
                            "Doctorate",
                            "Professional"])

degree_field = st.selectbox("Field of degree:",
                            ["Computer and information sciences",
                             "Mathematics and statistics",
                             "Agricultural and food sciences",
                             "Biological sciences",
                             "Environmental life sciences",
                             "Chemistry (except biochemistry)",
                             "Earth, atmospheric, and ocean sciences",
                             "Physics and astronomy",
                             "Other physical sciences",
                             "Economics",
                             "Political and related sciences",
                             "Psychology",
                             "Sociology and anthropology",
                             "Other social sciences",
                             "Aerospace, aeronautical and astronautical engineering",
                             "Chemical engineering",
                             "Civil and architectural engineering",
                             "Electrical and computer engineering",
                             "Industrial engineering",
                             "Mechanical engineering",
                             "Other engineering",
                             "Health",
                             "Science and mathematics teacher education",
                             "Technology and Technical Fields",
                             "Other science and engineering related fields",
                             "Management and administration fields",
                             "Education (other than science and math teacher education)",
                             "Social service and related fields",
                             "Sales and marketing fields",
                             "Art and humanities fields",
                             "Other non-science and engineering fields"])
employer = st.selectbox("Sector of employer:", 
                        ["Elementary, middle, or secondary school",
                         "2-year college, junior college, or technical institute",
                         "4-year college/univeristy",
                         "Medical School",
                         "University research institute",
                         "Other educational institution",
                         "Private for-profit (non-educational)",
                         "Private non-profit (non-educational)",
                         "Self-employed, not incorporated (non-educational)",
                         "Self-employed, incorporated (non-eduational)",
                         "Local government",
                         "State governemnt",
                         "US military",
                         "US government",
                         "Other non-educational"])
st.text("Location Key:\n"
"New England = Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont\n" \
"East North Central = Illinois, Indiana, Michigan, Ohio, Wisconsin\n" \
"West North Central = Iowa, Kansas, Minnesota, Missouri, Nabraska, N/S Dakota\n" \
"East South Central = Alabama, Kentucky, Tennessee, Mississippi\n" \
"West South Central = Arkansas, Louisiana, Oklahoma, Texas\n" \
"Mointain = Arizona, Colorado, Idaho, Montana, Nevada, New Mexico, Utah, Wyoming\n" \
"Pacific Region = Alaska, California, Oregon, Washington, Hawaii, other US territories")
loc_employer = st.selectbox("Location of employer:", 
                           ["New England",
                            "East North Central Region",
                            "West North Central Region",
                            "East South Central Region",
                            "West South Central Region",
                            "Mountain Region",
                            "Pacific Region",
                            "Mid-Atlantic Region",
                            "South Atlantic Region",
                            "US Territory (unspecified)",
                            "Europe (unspecified)",
                            "Northern Europe",
                            "Asia",
                            "South America",
                            "Americas (unspecified)"])
sector = st.selectbox("Employer sector:",
                      ["Educational Institution",
                       "Government",
                       "Business/Industry"])
job = st.selectbox("Job field:",
                   ["Computer and mathematical scientists",
                    "Biological, agricultural, and other life sciences",
                    "Physical and related scientists",
                    "Social and related scientists",
                    "Engineers",
                    "Other science and engineering occupations",
                    "Non-science and engineering occupations"])
independence = st.selectbox("How important is the job's degree of independence?",
                         ["Very important",
                          "Somewhat important",
                          "Somewhat unimportant",
                          "Not important at all"])
security = st.selectbox("How important is the job's security?",
                        ["Very important",
                          "Somewhat important",
                          "Somewhat unimportant",
                          "Not important at all"])
society = st.selectbox("How important is the job's contribution to society?",
                        ["Very important",
                          "Somewhat important",
                          "Somewhat unimportant",
                          "Not important at all"])
years = st.number_input("How many years will be between getting your bachelor's degree and getting your first job?",
                     min_value=0, max_value=5)






# Mappings
EMST_TOGA_mappings = {
    "New England" : "85",
    "East North Central Region" : "87",
    "West North Central Region" : "88",
    "East South Central Region" : "90",
    "West South Central Region" : "91",
    "Mountain Region" : "92",
    "Pacific Region" : "93",
    "US Territory (unspecified)" : "96",
    "Mid-Atlantic Region" : "97",
    "South Atlantic Region" : "98",
    "Europe (unspecified)" : "166",
    "Northern Europe" : "185",
    "Asia" : "249",
    "South America" : "374",
    #"Americas (unspecified)" : "399"
}
loc_code = EMST_TOGA_mappings[loc_employer]

RACETHM_mappings = {
    "Asian" : "1",
    "American Indian" : "2",
    "Native Alaskan" : "2",
    "Black" : "3",
    "Hispanic (any race)" : "4",
    "White" : "5",
    "Pacific Islander" : "6",
    "Multi-Racial" : "7"
}
race_code = RACETHM_mappings[race]

EMTP_mappings = {
    "Elementary, middle, or secondary school" : "1",
    "2-year college, junior college, or technical institute" : "2",
    "4-year college/univeristy" : "3",
    "Medical School" : "4",
    "University research institute" : "5",
    "Other educational institution" : "6",
    "Private for-profit (non-educational)" : "10",
    "Private non-profit (non-educational)" : "11",
    "Self-employed, not incorporated (non-educational)" : "12",
    "Self-employed, incorporated (non-eduational)" : "13",
    "Local government": "14",
    "State governemnt": "15",
    "US military" : "16",
    "US government":"17",
    "Other non-educational":"18"
}
employer_code = EMTP_mappings[employer]

EMSECSM_mappings = {
    "Educational Institution" : "1",
    "Government" : "2",
    "Business/Industry" : "3"
}

sector_code = EMSECSM_mappings[sector]

DGRDG_mappings = {
    "Bachelor's" : "1",
    "Master's" : "2",
    "Doctorate" : "3",
    "Professional" : "4"
}


degree_type_code = DGRDG_mappings[degree_type]

N2OCPRMG_mappings = {
    "Computer and mathematical scientists" : "1",
    "Biological, agricultural, and other life sciences" : "2",
    "Physical and related scientists" : "3",
    "Social and related scientists" : "4",
    "Engineers" : "5",
    "Other science and engineering occupations" : "6",
    "Non-science and engineering occupations" : "7"
}

job_code = N2OCPRMG_mappings[job]

NDGMENG_mappings = {
    "Computer and information sciences":"11",
    "Mathematics and statistics":"12",
    "Agricultural and food sciences":"21",
    "Biological sciences":"22",
    "Environmental life sciences":"23",
    "Chemistry (except biochemistry)":"31",
    "Earth, atmospheric, and ocean sciences":"32",
    "Physics and astronomy":"33",
    "Other physical sciences":"34",
    "Economics":"41",
    "Political and related sciences":"42",
    "Psychology":"43",
    "Sociology and anthropology":"44",
    "Other social sciences":"45",
    "Aerospace, aeronautical and astronautical engineering":"51",
    "Chemical engineering":"52",
    "Civil and architectural engineering":"53",
    "Electrical and computer engineering":"54",
    "Industrial engineering":"55",
    "Mechanical engineering":"56",
    "Other engineering":"57",
    "Health":"61",
    "Science and mathematics teacher education":"62",
    "Technology and Technical Fields":"63",
    "Other science and engineering related fields":"64",
    "Management and administration fields":"71",
    "Education (other than science and math teacher education)":"72",
    "Social service and related fields":"73",
    "Sales and marketing fields":"74",
    "Art and humanities fields":"75",
    "Other non-science and engineering fields":"76"
}

SURVEY_mappings = {
    "Very important":"1",
    "Somewhat important":"2",
    "Somewhat unimportant":"3",
    "Not important at all": "4"
}

degree_field_code = str(NDGMENG_mappings[degree_field])

independence_code = SURVEY_mappings[independence]
security_code = SURVEY_mappings[security]
society_code = SURVEY_mappings[society]








# Input
input_data = pd.DataFrame([{
    "AGE": age,
    "EMST_TOGA": loc_code,
    "RACETHM": race_code,
    "EMSECSM": sector_code,
    "DGRDG": degree_type_code,
    "NDGMENG": degree_field_code,
    "N2OCPRMG": job_code,
    "Years_To_First_Job": str(years),
    "EMTP": employer_code,
    "SEX_2023": sex,
    "FACIND": independence_code,
    "FACSEC": security_code,
    "FACSOC": society_code
}])

# Transform categorical columns as before


for col, le in encoders.items():
    input_data[col] = le.transform(input_data[col].astype(str))

# Reorder columns exactly like training
input_data = input_data[feature_order]







st.title("Your Results")

df = pd.read_csv("salary_train_data.csv")


# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Salary: ${prediction:,.0f}")
# Charting
    df_degree_field = df[df['NDGMENG'] == int(degree_field_code)]
    df_job_code = df[df['N2OCPRMG'] == int(job_code)]
    df_loc = df[df['EMST_TOGA'] == int(loc_code)]


    st.subheader("Your salary compared to others in your region")

    # Drop missing values
    values = df_loc["SALARY"].dropna()

    # Compute counts and bins
    counts, bins = np.histogram(values, bins=20)

    # Convert to percentages
    percentages1 = counts / counts.sum() * 100

    fig_loc, ax = plt.subplots()

    # Plot bar chart manually using bin widths
    ax.bar(
        bins[:-1],
        percentages1,
        width=np.diff(bins),
        edgecolor="black",
        align="edge",
        color="green",
    )

    # Add line for predicted salary
    ax.axvline(prediction, color="red", linestyle="--", linewidth=2, label="Your Salary")

    # Labels
    ax.set_xlabel("Salary")
    ax.set_ylabel("Percentage of people")
    ax.set_title(f"{loc_employer} Distribution")
    ax.legend()
    ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    st.pyplot(fig_loc)






    st.subheader("Your salary compared to others in your degree field")

    # Drop missing values
    values = df_degree_field["SALARY"].dropna()

    # Compute counts and bins
    counts, bins = np.histogram(values, bins=20)

    # Convert to percentages
    percentages2 = counts / counts.sum() * 100

    fig_deg, ax2 = plt.subplots()

    # Plot bar chart manually using bin widths
    ax2.bar(
        bins[:-1],
        percentages2,
        width=np.diff(bins),
        edgecolor="black",
        align="edge",
        color="green",
    )

    # Add line for predicted salary
    ax2.axvline(prediction, color="red", linestyle="--", linewidth=2, label="Your Salary")

    # Labels
    ax2.set_xlabel("Salary")
    ax2.set_ylabel("Percentage of people")
    ax2.set_title(f"{degree_field} Distribution")
    ax2.legend()
    ax2.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    st.pyplot(fig_deg)





    st.subheader("Your salary compared to others in your job field")

    # Drop missing values
    values = df_job_code["SALARY"]

    # Compute counts and bins
    counts, bins = np.histogram(values, bins=20)

    # Convert to percentages
    percentages3 = counts / counts.sum() * 100

    fig_job, ax3 = plt.subplots()

    # Plot bar chart manually using bin widths
    ax3.bar(
        bins[:-1],
        percentages3,
        width=np.diff(bins),
        edgecolor="black",
        align="edge",
        color="green",
    )

    # Add line for predicted salary
    ax3.axvline(prediction, color="red", linestyle="--", linewidth=2, label="Your Salary")

    # Labels
    ax3.set_xlabel("Salary")
    ax3.set_ylabel("Percentage of people")
    ax3.set_title(f"{job} Distribution")
    ax3.legend()
    ax3.ticklabel_format(style='plain', axis='x')
    ax3.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    st.pyplot(fig_job)




# Notes
st.subheader("Model Notes")

st.text("The model was trained on data from the 2023 National Survey of College Graduates database.\n" \
"The data was filtered to highlight only graduates getting their first job out of college by:\n" \
"Filtering for those who have a BA and started their job that year or later\n" \
"Filtering for those whose job requires expertise from thier degree\n" \
"Filtering for those who got their first job 5 or less years after getting their BA or 10 or less years for those with advanced degrees\n" \
"Removing outliers such as those who were above the age of 36 and had a salary outside of the range from $15,000 to $600,000")

st.text("The NSCG database is heavily weighted towards those in science and engineering fields; users in fields outside these may experience increased error in results.")
st.text("For more information about the survey, visit this website:")
st.link_button(label="NSCG 2023 Survey",url="https://ncses.nsf.gov/surveys/national-survey-college-graduates/2023")

st.subheader("Survey")
st.text("Please take this quick survey to let us know about your experience!")
st.link_button(label="Feedback",url="https://forms.office.com/Pages/ResponsePage.aspx?id=2RNYUX1x3UWeypqhnAnW-SVikx1a_l9DriBOVBbK_StUNkM3SUZZMlFVVUdJTUlaWTVGR1JKVlZVRS4u")





