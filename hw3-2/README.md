# HW3-2 2D SVM

* This is a homework for course "AIoT-DA", which is a 2D SVM demo.
* This program is made by ChatGPT

## Demo

![demo.gif](demo.gif)

## Prompt

1. Raw prompt

```plaintext
Write python to implement a system to perform SVM on a set of 300 randomly generated variables, and visualize the results. Here are some requirements:
1. 300 random data with 2 feature(x,y) and two category(c1,c2)
2. if |x|<500 and |y|<500 then is c1, else is c2
3. demonstrate the plot for SVM predicating result and predicate boundary.
```

2. Generated prompt

```plaintext
You’re a highly skilled Python developer with extensive experience in implementing machine learning algorithms, particularly Support Vector Machines (SVM). You have a deep understanding of data visualization techniques and how to effectively communicate the results of complex models.

Your task is to write a Python script that implements an SVM on a set of 300 randomly generated variables and visualizes the results. 

Here are the specific requirements for the task:  
1. Generate 300 random data points with two features (x, y) and two categories (c1, c2).  
2. Assign the category c1 to points where |x| < 500 and |y| < 500; otherwise, assign category c2.  
3. Include a demonstration of the plot showing the SVM predicting results and the decision boundary.

Make sure to use appropriate libraries such as NumPy, Matplotlib, and Scikit-learn to achieve this. Additionally, provide clear comments in the code to facilitate understanding.
```

3. Follow up question

```plaintext
Here are two modify.
1. Now use streamlit to host the program. I want to control the number of random data and the label boundary.
2. Draw a new plot in 3D. I want to see the hyper plane in 3D.
```

4. Follow up question for debugging

```plaintext
I want to demonstrate the hyperplane but got error:coef_ is only available when using a linear kernel.
```