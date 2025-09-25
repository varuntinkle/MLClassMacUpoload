# How to Install
Using UV let's prepare our environment first. UV will install all our dependencies.
```ps
uv sync
```
You can try to install pytorch with a GPU backend, if you have one available. 
```ps
uv pip install torch --torch-backend=auto
```

To run the Python Notebooks, we can use the Jupyter Lab. Simply start the lab with

```ps
uv run --with jupyter jupyter lab
``` 

# How to Code
For this project we will focus on the components we will regularly use while developing and training networks. Take a look in the files in `layers/`. 

You will need to fill the following sections in each file.
```python
##############################
# YOUR CODE HERE

##############################
```

# Assignments
Here are the explanations of the tasks for this project.

- [Linear Layer Implementation](task_descriptions/LinearLayer.md)


# How to Test
You can test your progress by simply running `uv run pytest .\tests\  -v` in the homework folder. The goal of this project is to pass all the tests by implementing the necessary parts in our `layers\` folder.