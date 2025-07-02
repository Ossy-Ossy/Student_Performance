import numpy as np 
import pandas as pd
import joblib
import streamlit as st


x = pd.read_csv("x_student_performance.csv")
x.drop(x.columns[0] ,axis = 1,inplace = True)

df = pd.read_csv("student.csv")
df.columns = df.columns.str.strip()

model = joblib.load("model_student.joblib")
scaler = joblib.load("scaler_student.joblib")

st.write("""
# Welcome to the AI-Powered Student Performance Prediction System   

This app uses student data to predict their performance in an examination.      
""")

st.write("***")

# build prediction system
def student_prediction(x,model,scaler,name,study_hours,attendance_rate ,past_exam_score,
                       gender,parental_education,internet_access,Extracurricular):
    fv = np.zeros(len(x.columns))
    study_hours,attendance_rate,past_exam_score = float(study_hours),float(attendance_rate),float(past_exam_score)
    obj_cols = [gender, parental_education ,internet_access,Extracurricular]
    scaled_values = scaler.transform(np.array([[study_hours,attendance_rate,past_exam_score]]))[0]
    fv[0:3] = scaled_values
    for cols in obj_cols:
        index = np.where(df.columns == cols)[0]
        if index.size > 0:
            fv[index] = 1
    fv = fv.reshape(1,-1)
    prediction = model.predict(fv)[0]
    if prediction >= 60:
        return f"{name} will pass with a mark of {prediction.round()}"
    else:
        return f"{name} will not pass with a mark of {prediction.round()}"

name = st.text_input("What is the name of the student??")

gender = st.selectbox("Select sex of student" ,df['Gender'].unique())

study_hours = st.number_input("How many hours do you study per week??",
min_value = df['Study_Hours_per_Week'].min()
)

parental_education = st.selectbox("What is the level of the student's parental education??" ,df['Parental_Education_Level'].unique())

attendance_rate = st.number_input("What is the student's attendance rate(percentage%)",
min_value = df['Attendance_Rate'].min() ,max_value = df['Attendance_Rate'].max()
)

internet_access = st.selectbox("What is the satus of internet access of the student" , x.columns[9:11])

past_exam_score = st.number_input("What is the past exams/test score of the student??",
min_value = df['Past_Exam_Scores'].min() ,max_value = df['Past_Exam_Scores'].max()
)

Extracurricular = st.selectbox("Does the student partake in extracurricular activities", x.columns[11:])

st.sidebar.header("🎓 Key Success Factors")

st.sidebar.write("""
**Critical Indicators for Passing:**  
- **Attendance Rate** (>85%): Higher attendance correlates with better grades.  
- **Study Hours** (≥ 30 hrs/week): Consistent study time improves understanding.  
- **Past Exam Scores** (>70%): Strong past performance predicts future success.  
- **Extracurricular Balance**: Moderate involvement (neither too high nor too low).  
- **Internet Access**: Essential for research and learning resources.  
""")

st.sidebar.header("📈 Boost Your Performance")

st.sidebar.write("""
If the model predicts **"Poor" or "Fail"**, focus on:  
1. **📅 Attendance**: Aim for **>90%** class participation.  
2. **⏳ Study Hours**: Increase to **≥ 35 hrs/week** (e.g., 3 hrs/day).  
3. **📝 Past Exams**: Analyze mistakes, retake practice tests.  
4. **🔍 Parental/Teacher Support**: Seek help if struggling.  
5. **🌐 Internet Access**: Leverage online learning tools (Khan Academy, Coursera).  

*Small consistent efforts compound over time!*  
""")

st.sidebar.header("🔧 Model Insights")

st.sidebar.write("""
This AI predicts outcomes using:  
- **Study Habits** (Hours/Attendance)  
- **Academic History** (Past Scores)   
- **Environment** (Parental Education, Internet Access)  
- **Balance** (Extracurricular Activity)  

*Trained on educational datasets to identify success patterns.*  
""")

st.sidebar.header("✨ Success Starts Here")

st.sidebar.write("""
Improving these metrics helps:  
- Avoid course repetition/failure.  
- Build discipline for future careers.  
- Unlock scholarships/higher education.  
- Parents/teachers intervene early.  
""")
if st.button('Predict Performance'):
    result = student_prediction(x,model,scaler,name,study_hours,attendance_rate ,past_exam_score,
                                gender,parental_education,internet_access,Extracurricular)
    st.success(result)
