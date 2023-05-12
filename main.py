import pickle
import numpy as np
import pandas as pd
import KNN
import ontology_manager, RandomForest, SupportVectorMachine, KMeans, CSP_AnalysisPrenotation, logisticRegression, neuralNetwork



def main():
    print("Preparazione dati...")
    #preparazione dati 
    feature = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "bladder", "bladder-quad", "irradiat"]
    feature_dummied = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "bladder", "bladder-quad", "irradiat"]
    dataset = pd.read_csv("breast-cancer.csv", sep=",", names=feature, dtype={'class':object, 'age':object, 'menopause':object, 'tumor-size':object, 'inv-nodes':object, 'node-caps':object, 'deg-malig':np.int32, 'bladder': object, 'bladder-quad':object, 'irradiat':object})
    data_dummies = pd.get_dummies(dataset, columns=feature_dummied)
    data_dummies = data_dummies.drop(["class"], axis=1)

    x = data_dummies
    y = pd.get_dummies(dataset["class"], columns=["class"])
    y = y["recurrence-events"]
    print("Completata!")
    input("Press Enter to continue...")

    while True:
        print("Seleziona un'opzione:")
        print("1. Esegui classificazione supervisionata")
        print("2. Esegui classificazione non supervisionata")
        print("3. Informazioni sulla patologia")
        print("4. Prenota una visita istologica")
        print("0. Esci")
        choice = input("Inserisci il numero corrispondente all'opzione desiderata: ")

        if choice == "1":
            print("Algoritmo: K-Nearest Neighbors")
            KNN.KNearestNeighbors(x, y)
            input("Premi Invio per continuare...")

            print("ALGORITMO: Random Forest")
            RandomForest.RF(x,y)
            input("Premi Invio per continuare...")

            print("ALGORITMO: Logistic Regression ")
            logisticRegression.logistic_regression(x,y)
            input("Premi Invio per continuare...")

            print("ALGORITMO: Support Vector Machine")
            SupportVectorMachine.SVM(x,y)
            input("Premi Invio per continuare...")

            print("Rete Neurale")
            neuralNetwork.neural_network(x,y)
            input("Premi Invio per continuare...")

        elif choice == "2":
            print("ALGORITMO: K-Means")
            KMeans.KMEANS(x,y)

        elif choice == "3":
            print("\n\nInformazioni sulla patologia\n")
            ontology_manager.ontology_analyzer()

        elif choice == "4":
            CSP_AnalysisPrenotation.lab_booking()

        elif choice == "0":
            print("Uscita...")
            break

        else:
            print("Scelta non valida. Inserisci un numero corrispondente all'opzione desiderata.")
   
main()