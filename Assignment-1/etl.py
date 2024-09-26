import psycopg2
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pymongo import MongoClient

# PART 1: EXTRACTION

conn=psycopg2.connect(
    dbname="collegecourses",
    user="postgres",
    password="livingdaylights",
    host="localhost",
    port="5432"
)
tables=['students','instructors','departments','courses','corecourses','enrollments']
df={}
for i in tables:
    query=f"select * from {i};"
    df[i]=pd.read_sql_query(query,conn)
print(df)
conn.close()
print("Records Extracted Successfully")

#PART 2: TRANSFORMATION

students_data=df['students'][['student_id','first_name','last_name','department_id','yoe','email']].to_dict(orient='records')
instructors_data=df['instructors'][['instructor_id','first_name','last_name','email','department_id']].to_dict(orient='records')
departments_data=df['departments'][['department_id','department_name']].to_dict(orient='records')
courses_data=df['courses'][['course_id','course_name','department_id','course_semester','credits']].to_dict(orient='records')
corecourses_data=df['corecourses'][['department_id','course_id']].to_dict(orient='records')
enrollments_data=df['enrollments'][['enrollment_id','student_id','course_id','instructor_id']].to_dict(orient='records')
print("Records Transformed Successfully")

#PART 3: LOADING

mdb=MongoClient('mongodb://localhost:27017/')['collegecourses']
mdb.students.insert_many(students_data)
mdb.instructors.insert_many(instructors_data)
mdb.departments.insert_many(departments_data)
mdb.courses.insert_many(courses_data)
mdb.corecourses.insert_many(corecourses_data)
mdb.enrollments.insert_many(enrollments_data)
print("Records Migrated")
