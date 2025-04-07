1. Setting the virtual environment

First, within the project folder, we run the following command to create a virtual environment:
python -m venv ./env
pip install -e .
which python
python -v


2. Setup packages

I also may want to access some classes without calling the files, 
so I may want to make it easier on myself and have them accessible at the top of the project.
Example see setup.py at the root of the project

Run the following at the root of the project:
pip install -e .
Now in a jupyter notebook, we can run the command
from ml_serevice import DataProcessor

This only works in the virtual environment, though. 
We will look at more general uses of the package in a later article.


## Reference

https://www.dailydoseofds.com/how-to-structure-your-code-for-machine-learning-development/?utm_source=substack&utm_medium=email
