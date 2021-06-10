This is django app which dumps dialogs from pandas dataframe representation

TODO specify how to get the representation

upload_dialogs_to_db.py - loads all dialogs into database
 (you need to have pickled dataframe with ratings)

python manage.py migrate
python manage.py makemigrations
python manage.py createsuperuser
python manage.py runserver


python upload_dialogs_to_db.py


See also:
http://192.168.10.188:8081/index.php/Alexa_Prize_Social_Bot#.D0.9D.D0.B0.D1.81.D1.82.D1.80.D0.BE.D0.B9.D0.BA.D0.B0_postgresql