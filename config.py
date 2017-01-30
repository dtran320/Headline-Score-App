import os

WTF_CSRF_ENABLED = True
SECRET_KEY = ""
DEBUG = True
if 'DYNO' in os.environ:
  SECRET_KEY = os.environ.get("SECRET_KEY")
else:
  SECRET_KEY = "never-guess"
