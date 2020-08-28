"""
Performs analysis on the correlation between university ratings and
admission rates by building a regression model and creating plots.
"""


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def university_rating_analysis(data):
    """
    Analyzes how university ratings affect admission rate with various plots
    as well as a regression model.
    """
    # Split data into training set and testing set
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Make a plot to see the general trend of data
    plt.scatter(train_data['University Rating'],
                train_data['Chance of Admit'], marker='+', label='Train')
    plt.scatter(test_data['University Rating'],
                test_data['Chance of Admit'], marker='.', label='Test')
    plt.legend()
    plt.xlabel('University Rating')
    plt.ylabel('Chance of Admit')
    plt.title('Visualization of Dataset')
    plt.savefig('dataset_visualization.png')
    plt.clf()

    # Train the linear regression model
    model = LinearRegression().fit(train_data[['University Rating']],
                                   train_data['Chance of Admit'])
    predict_val = model.predict(test_data[['University Rating']])
    score = model.score(train_data[['University Rating']],
                        train_data['Chance of Admit'])
    print('Model score: ' + str(score))

    # Plot outputs
    plt.scatter(test_data['University Rating'], test_data['Chance of Admit'],
                color='black')
    plt.plot(test_data['University Rating'], predict_val, color='blue',
             linewidth=3)
    plt.xlabel('University Rating')
    plt.ylabel('Chance of Admit')
    plt.title('Regression Model VS True Values')
    plt.savefig('university_rate_model.png')
