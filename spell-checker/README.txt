Spell checker

The purpose of this model is to suggest the most probable correct options to a wrongly spelled word by the user.

Python files:
utils.py >> helper file to get the methods called from execute.py and execute_lite.py
execute.py >> main file that will read the content, build some dictionaries and serve as spell checker for the given input
execute_lite.py >> alternative version to execute.py with loaded pickles instead of building everything from scratch

NOTE: if you haven't built the SQL database yet (it's done automatically by running the execute.py file), you'll need to place a TMX file, or a set of TMX files, into a folder "dataset" in the same directory and set the variable convert_tmx_to_sql to True (line 11 of execute.py). If you have already built the SQL database, you can skip this task (in order not to reimport into the database the same stuff) by setting the variable convert_tmx_to_sql to False (again, line 11 of execute.py).
More information below.

The "dataset" directory should include either a TMX file (or a set of TMX files) or a database. For this task, I used a set of 1,357 TMX files (more than 4 million translation units in many language pairs) and it resulted in a 1GB database, but you can use the resources you would like. I am leaving this folder empty because GitHub has a limit for file uploads, but the TMX files used for this test can be found here: http://wt-public.emm4u.eu/Resources/DGT-TM-2019/Vol_2018_1.zip. After downloading the ZIP, unzip it and place it under a "dataset" directory (the TMX files can be under the folder "Vol_2018_1" since the script will read the subfolders too), make sure the convert_tmx_to_sql variable is set to True (line 11 of execute.py) and run the file. For the next usages, if you are not importing anything else and you just want to try out your model, set the convert_tmx_to_sql variable to False (again, line 11 of execute.py) so the content is not re-imported.

If you prefer not to download the huge amount of data, I have saved as a pickle the Python probability that will be used to get the suggestions for the model, as well as the vocabulary list. These can be found under pickled/probs.pickle and pickled/vocab.pickle, and they can be used in execute_lite.py. So if you'd like to download the original files and build everything from scratch, follow the steps from the previous paragraph and use execute.py. If you just want to use the model with the pickles created out of that process, you can use execute_lite.py.

NOTE: the TMX files are currently modified before being parsed for a better performance, so keep a copy of these before processing them.
