# Titanic-Dataset-Regression

Random Forest Regression on the Titanic Dataset demonstrating the use of the Random Forest Regression model.

We use `sklearn` library to model the machine learning behavior and we focus on predicting survivability based on the titanic dataset.

###### We end up getting a C-Stat score of 87.1%

## Optimization

The Random Forest Regression model can be optimizaed using `n_estimators`, `max_features`, and `min_sample_leafs`

We see the following results:

## N-Estimator Optimization (100)

![N-Estimator](/img/opt_estimator.png)

## Max-Features Optimization (Auto)

![Max-Features](/img/opt_features.png)

## Min-Leaf Optimization (5)

![Min-Leaf](/img/opt_min_leaf.png)


#Conclusion

Our final model therefore has the highest accuracy of 87.1% with 100 Estimators, all features, and 5 minimum samples.
