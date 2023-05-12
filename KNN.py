import pickle
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def KNearestNeighbors(X, y):

    X_train, X_test, y_train, y_test = train_test_split (X, y, train_size=0.75, random_state=13)
    print(X_test)

    error = []
    #calcolo errore per K tra 1 e 20
    for i in range(1,20):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    #grafico che mostra l'errore medio nelle predizioni a seguito di una variazione del valore K (numero vicini)
    plt.plot(range(1, 20), error, color='red', linestyle = 'dashed', marker = 'o', markerfacecolor='blue', markersize = 10)
    plt.title('Error Rate K value')
    plt.xlabel('K value')
    plt.ylabel('Mean Error')
    plt.show()

    neigh = KNeighborsClassifier(n_neighbors=2)
    knn = neigh.fit(X_train, y_train)

    prediction = knn.predict(X_test)
    accuracy = accuracy_score(prediction, y_test)

    print('Classification report \n', classification_report(y_test, prediction))

    #train modello con cv di 5
    cv_scores = cross_val_score(neigh, X, y, cv=5)

    print('\ncv_scores mean:{}'.format(np.mean(cv_scores)))
    print('\ncv_score variance:{}'.format(np.var(cv_scores)))
    print('\ncv_score dev standard:{}'.format(np.std(cv_scores)))

     #AUC
    probs = knn.predict_proba(X_test)
    # mantengo solo le probabilità positive
    probs = probs[:, 1]

    auc = roc_auc_score(y_test, probs)
    print('AUC: %.3f' % auc)
    # calcolo roc curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plot
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('FP RATE')
    pyplot.ylabel('TP RATE')
    
    confusion_Matrix = confusion_matrix(y_test, prediction)

    df_cm = pd.DataFrame(confusion_Matrix, index = [i for i in "01"], columns = [i for i in "01"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)

    pyplot.show()

    average_precision = average_precision_score(y_test, prediction)
    precision, recall, _ = precision_recall_curve(y_test, prediction)
    f1 = f1_score(y_test, prediction)


    # plot
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    
    #ristampo i dati di interesse
    print('accuracy, average precision and f1-score are:', accuracy, average_precision, f1)

    #salvo il modello su disco per uso futuro
    pickle.dump(knn, open("MODEL_knn.sav", 'wb'))