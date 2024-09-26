import matplotlib.pyplot as plt
from pyspark.sql.functions import countDistinct
from pyspark.sql import SparkSession
import time
# import pyspark
# print(pyspark.__version__)
plotarr=[]
# Spark Session Created
spark = SparkSession\
    .builder \
    .appName("MongoDBIntegration") \
    .config("spark.mongodb.read.connection.uri","mongodb://localhost:27017/collegecourses") \
    .config("spark.mongodb.write.connection.uri","mongodb://localhost:27017/collegecourses") \
    .config("spark.jars.packages","org.mongodb.spark:mongo-spark-connector_2.12:10.4.0") \
    .getOrCreate()

#Loading Data

students_df=spark.read.format("mongodb").option("collection","students").load()
instructors_df=spark.read.format("mongodb").option("collection","instructors").load()
departments_df=spark.read.format("mongodb").option("collection","departments").load()
courses_df=spark.read.format("mongodb").option("collection","courses").load()
corecourses_df=spark.read.format("mongodb").option("collection","corecourses").load()
enrollments_df=spark.read.format("mongodb").option("collection","enrollments").load()
enrollments_df=enrollments_df.repartitionByRange("enrollment_id")
students_df.show() #Helps showcase that the program is working plus eliminates the cold start nonsense

#Question-1 (All Students enrolled in a specific course)
course_id=1  #C++
a=time.time()
enrolled_students = enrollments_df.filter(enrollments_df.course_id==course_id) \
                                    .join(students_df,"student_id") \
                                    .select(students_df.first_name,students_df.last_name)
print("Students Enrolled in Course with Course ID ",course_id, " -")
enrolled_students.show()
b=time.time()
print(b-a, "seconds")
plotarr.append(b-a)


#Question-2 (Average Count students enrolled in courses offered by a particular instructor)
a=time.time()
courses_with_instructors=courses_df.join(instructors_df,"department_id")
instructor_id=1 #Walter White
enrollments_for_instructor=enrollments_df.join(courses_with_instructors,"course_id").filter(instructors_df.instructor_id==instructor_id)
avg_students_per_course=enrollments_for_instructor.groupBy("course_id").count().agg({"count": "avg"})
print("Average Number of Students Enrolled in Course with Instructor ID ",instructor_id, " -")
avg_students_per_course.show()
b=time.time()
print(b-a, "seconds")
plotarr.append(b-a)


#Question-3 (Courses Offered by a Specific Department)
a=time.time()
department_id=100 #Computer Science
courses_by_department=courses_df.filter(courses_df.department_id==department_id)\
    .select(courses_df.course_name)
print("Courses Offered by a Specific Department ",department_id, " -")
courses_by_department.show()
b=time.time()
print(b-a, "seconds")
plotarr.append(b-a)

#Question-4 (Students per Department)
a=time.time()
students_per_dept=students_df.groupBy("department_id").count()
print("Students per Department -")
students_per_dept.show()
b=time.time()
print(b-a, "seconds")
plotarr.append(b-a)

#Question-5 (Instructors Who Have Taught All CSE Core Courses)
a=time.time()
cse_corecourses_df=corecourses_df.filter(corecourses_df.department_id==100).select("course_id")
enrollments_for_cse_core_df=enrollments_df.join(cse_corecourses_df,"course_id").select("instructor_id","course_id")
instructor_core_course_count_df=enrollments_for_cse_core_df.groupBy("instructor_id").agg(countDistinct("course_id").alias("core_courses_taught"))
total_cse_core_courses=cse_corecourses_df.count()
instructors_taught_all_cse_df=instructor_core_course_count_df.filter(instructor_core_course_count_df.core_courses_taught==total_cse_core_courses)
instructors_final_df=instructors_taught_all_cse_df.join(instructors_df,"instructor_id")
print("Instructors Who Have Taught All CSE Core Courses -")
instructors_final_df.show()
b=time.time()
print(b-a, "seconds")
plotarr.append(b-a)

#Question-6 (Top 10 Courses with the Highest Enrollments)
a=time.time()
top_courses=enrollments_df.groupBy("course_id").count().orderBy("count",ascending=False).limit(10)
print("Top 10 Courses with the Highest Enrollments -")
top_courses.show()
b=time.time()
print(b-a, "seconds")
plotarr.append(b-a)
print(plotarr)
plt.plot([1,2,3,4,5,6],plotarr)
plt.xlabel("Queries")
plt.ylabel("Time taken per query in seconds")
plt.show()
