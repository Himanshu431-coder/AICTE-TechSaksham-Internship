# AI Based Fitness Calorie Tracker Using Python (AICTE Internship Project)


# AI-Based Fitness Calorie Tracker

## Project Overview
The **AI-Based Fitness Calorie Tracker using Python** is an innovative application developed as part of my internship submission for the All India Council for Technical Education (AICTE). This project leverages artificial intelligence to assist users in tracking their calorie intake based on their fitness activity data. By employing advanced machine learning models, the application aims to provide accurate predictions of calories burned during various physical activities.

## Key Features
- **Calorie Prediction**: Utilizes sophisticated machine learning algorithms to predict calorie expenditure based on user-input fitness activity data.
- **Interactive Web Application**: Developed using Streamlit, this feature allows users to seamlessly input their fitness data and receive real-time predictions regarding calories burned.
- **Data Visualization**: Employs Matplotlib and Seaborn for visualizing critical data trends, enhancing users' understanding of their calorie consumption and exercise performance.

## Technologies Used
- **Programming Language**: Python  
- **Development Environment**: Jupyter Notebook  
- **Web Framework**: Streamlit  
- **Data Manipulation Libraries**: NumPy, Pandas  
- **Data Visualization Libraries**: Matplotlib, Seaborn  
- **Machine Learning Library**: Scikit-learn  

---
## Machine Learning Techniques

- **Linear Regression**: Implemented to predict continuous values such as calories burned based on user activity.
- **Random Forest Regressor**: Employed to enhance prediction accuracy through the aggregation of multiple decision trees.
- **Grid Search Cross-Validation (CV)**: Utilized for optimizing hyperparameters, thereby improving overall model performance.
- **XGBoost Regressor (XGBRegressor)**: A powerful gradient boosting model that builds an ensemble of trees sequentially to improve predictions, offering high accuracy and handling missing data effectively.

- # 💡 Need of the model:

There are many apps that track the amount of calories you consume, but few that calculate the number of calories burned during a workout. This model can be developed as a standalone app to calculate calories burned, or it can be integrated into existing apps, such as HealthifyMe, to provide users with a comprehensive view of both the calories consumed and the calories burned throughout the day. This would give users a complete insight into their diet and overall health.

### 📚 About the dataset:

the csv has 4 datasets exercise, calories, workout and workout_logs

### 📕 exercise dataset:-

this csv contains all the info on the person's workout (has our independet features)

| #   | Column     | Non-Null Count | Dtype   |
|:---: | :------: | :--------------: | :-----: |
| 0   | User_ID    | 15000 non-null | int64   |
| 1   | Gender     | 15000 non-null | object  |
| 2   | Age        | 15000 non-null | int64   |
| 3   | Height     | 15000 non-null | float64 |
| 4   | Weight     | 15000 non-null | float64 |
| 5   | Duration   | 15000 non-null | float64 |
| 6   | Heart_Rate | 15000 non-null | float64 |
| 7   | Body_Temp  | 15000 non-null | float64 |

the Column are self-Exploratory

### 📗 calories dataset:-

this csv contains calories burned by all workouts in exercise (has our target value / dependent feature "calories")

| #   | Column    | Non-Null Count  | Dtype    | 
| :---:  | :------: | :--------------: | :-----: | 
| 0   | User_ID   | 15000 non-null  | int64    |
| 1   | Calories  | 15000 non-null  | float64  |

1. User_ID: unique id for each person
2. Calories: the the total calories they burned during their workout

### 📙 workout:-

* this is our main csv, this is a combination of all Columns from
exercise (features) and calories burned in each workout (target)

* both calorie_exercise_workout_model_analysis.ipynb and main.pyw
use this csv for training their models


| #  |  Column     |  Non-Null Count | Dtype   |
|:---: |  :------: |  :--------------: | :-----: |
| 0  |  User_ID    |  15000 non-null | int64   |
| 1  |  Gender     |  15000 non-null | object  |
| 2  |  Age        |  15000 non-null | int64   |
| 3  |  Height     |  15000 non-null | float64 |
| 4  |  Weight     |  15000 non-null | float64 |
| 5  |  Duration   |  15000 non-null | float64 |
| 6  |  Heart_Rate |  15000 non-null | float64 |
| 7  |  Body_Temp  |  15000 non-null | float64 |
| 8  |  Calories   |  15000 non-null | float64 |

this is a combination of exercise and calories csv

### 📓 workout_logs:-

* this csv just saves the workout info with calories burnt for the purpose of keeping track and logging

# 🖋️ Insights

### 1. ratio of male to female

![image](https://user-images.githubusercontent.com/91218998/223949818-621a734a-e112-4a9f-a7c3-5788eeee44c3.png)

the ratio of male to female is almost equal, but the number of females is a little bit more

### 2. Age range of people 

![image](https://user-images.githubusercontent.com/91218998/223950036-3b2dc960-7812-435b-8b1a-174aa2c8b4b9.png)


### there are many more insights i have grabbed from the dataset they are saved in the calorie_tracker.ipynb


--- 
## How to Clone and Run the Project:

### Prerequisites
To ensure a smooth setup and execution of the project, please ensure that you have the following installed:
- Python (preferably version 3.7 or higher)
- Jupyter Notebook
- Streamlit
- Required Python libraries (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn)

### Steps to Clone and Run the Project

1. **Clone the Repository**:  
   Open your terminal or command prompt and execute the following command:
   ```bash
   git clone https://github.com/Himanshu431-coder/AICTE-TechSaksham-Internship
   ```

2. **Navigate to the Project Directory**:  
   After cloning the repository, change your directory to the project folder:
   ```bash
   cd myproject
   ```
   
3. **Activate the virtual enviornment**:
   ```bash
   scripts\activate
   ```
 
4. **Install Required Dependencies**:  
   Install the necessary Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Jupyter Notebook**:  
   Launch Jupyter Notebook to execute the project code:
   ```bash
   jupyter notebook
   ```

6. **Launch the Streamlit Web Application**:  
   To initiate the interactive web application, use the following command:
   ```bash
   streamlit run fit_track.py
   ```

7. **View the Results**:  
   Once the Streamlit app is operational, it will automatically open in your web browser where you can input your data and monitor your fitness calorie tracking.

 # 🪟 Screenshots of inputs and outputs of fit_track.py :


### 1.asking for height, weight, age, gender 

![image](https://github.com/Himanshu431-coder/AICTE-TechSaksham-Internship/blob/main/Screenshot%20(24).png)


### 2. asking for body temprature, duration of the exercise, avarage heart rate during the exercise 1

![image](https://user-images.githubusercontent.com/91218998/223958388-9ca811d5-483a-4174-a2a6-145f28387e36.png)

### 3. calories burnt in exercise 1 (output)

![image](https://user-images.githubusercontent.com/91218998/223958900-961b8b7d-7024-40a5-81fa-12e5800f248a.png)

### 4. asking for body temprature, duration of the exercise, avarage heart rate during the exercise 2

![image](https://user-images.githubusercontent.com/91218998/223959166-1a2f05e2-7a66-48f3-b361-5ec3f7afa285.png)

### 5. calories burnt in exercise 2

![image](https://user-images.githubusercontent.com/91218998/223959345-d013170a-ea3d-4c93-85eb-7033d07dd8c8.png)

### 🖨️ 6. final output

![image](https://user-images.githubusercontent.com/91218998/223959498-90ea2783-398b-45b7-822e-4b4cabb05e38.png)

### workout_logs csv after saving the workout

![image](https://user-images.githubusercontent.com/91218998/225088450-af63292d-6239-49d1-b504-615b435ee0eb.png)

  

---
## Internship Mentor
This project was conducted under the mentorship of **Mr. Saomya Chaudhary**, whose expertise in machine learning and data science has significantly enriched my learning experience. His guidance facilitated my understanding of practical applications of machine learning algorithms, model optimization techniques, and effective data visualization strategies.

## Learning Outcomes
Throughout this internship, I acquired practical skills in implementing machine learning algorithms such as Linear Regression and Random Forest. I gained hands-on experience in essential processes including data preprocessing, feature engineering, and hyperparameter tuning using Grid Search CV. Furthermore, I successfully integrated a machine learning model into an interactive web application using Streamlit. Additionally, I honed my analytical skills by utilizing Python libraries such as Pandas and NumPy for data manipulation, and effectively visualizing results with Matplotlib and Seaborn. This internship has been instrumental in bridging the gap between theoretical knowledge and practical application.

---

## Acknowledgements
I would like to express my sincere gratitude to Mr. Saomya Chaudhary for his invaluable mentorship throughout this internship. His insights have been instrumental in my development as a data scientist. 

Additionally, I extend my appreciation to the libraries and frameworks that facilitated this project’s implementation, including NumPy, Pandas, Streamlit, and Scikit-learn.

---






