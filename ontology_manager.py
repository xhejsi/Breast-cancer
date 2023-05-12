from SPARQLWrapper import SPARQLWrapper, JSON

def ontology_analyzer():
    # Imposta l'endpoint di DBpedia
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    while True:
        # Chiede all'utente di scegliere un'opzione
        print("Scegli un'opzione:")
        print("1. Descrizione")
        print("2. Sintomi")
        print("3. Trattamenti")
        print("4. Fattori di rischio")
        print("0. Esci")
        choice = input("Scelta: ")

        if choice == "1":
            # Imposta la query sparql per recuperare la descrizione del cancro al seno
            sparql.setQuery("""
                PREFIX dbo: <http://dbpedia.org/ontology/>
                SELECT ?abstract
                WHERE {
                    <http://dbpedia.org/resource/Breast_cancer> dbo:abstract ?abstract .
                    FILTER (lang(?abstract) = "en")
                }
            """)

            # Imposta il formato di output dei risultati
            sparql.setReturnFormat(JSON)

            # Esegue la query sparql e recupera i risultati
            results = sparql.query().convert()

            # Stampa la descrizione recuperata
            print("Descrizione:")
            for result in results["results"]["bindings"]:
                abstract = result["abstract"]["value"]
                print(abstract)

        elif choice == "2":
            # Imposta la query sparql per recuperare i sintomi del cancro al seno
            sparql.setQuery("""
                PREFIX dbo: <http://dbpedia.org/ontology/>
                SELECT ?symptoms
                WHERE {
                    <http://dbpedia.org/resource/Breast_cancer> dbp:symptoms ?symptoms .
                }
            """)

            # Imposta il formato di output dei risultati
            sparql.setReturnFormat(JSON)

            # Esegue la query sparql e recupera i risultati
            results = sparql.query().convert()

            # Stampa i sintomi recuperati
            print("Sintomi:")
            for result in results["results"]["bindings"]:
                symptoms = result["symptoms"]["value"]
                print(symptoms)

        elif choice == "3":
            # Imposta la query sparql per recuperare i trattamenti del cancro al seno
            sparql.setQuery("""
                PREFIX dbo: <http://dbpedia.org/ontology/>
                SELECT ?treatment
                WHERE {
                    <http://dbpedia.org/resource/Breast_cancer> dbp:treatment ?treatment .
                }
            """)

            # Imposta il formato di output dei risultati
            sparql.setReturnFormat(JSON)

            # Esegue la query sparql e recupera i risultati
            results = sparql.query().convert()

            # Stampa i trattamenti recuperati
            print("\nTrattamenti:")
            for result in results["results"]["bindings"]:
                treatment = result["treatment"]["value"]
                print(treatment)

        elif choice == "4":
            # Imposta la query sparql per recuperare i fattori di rischio del cancro al seno
            sparql.setQuery("""
                PREFIX dbo: <http://dbpedia.org/ontology/>
                SELECT ?risks
                WHERE {
                    <http://dbpedia.org/resource/Breast_cancer> dbp:risks ?risks .
                }
            """)

            # Imposta il formato di output dei risultati
            sparql.setReturnFormat(JSON)

            # Esegue la query sparql e recupera i risultati
            results = sparql.query().convert()

            # Stampa i fattori di rischio recuperati
            print("\nFattori di rischio:")
            for result in results["results"]["bindings"]:
                risks = result["risks"]["value"]
                print(risks)

        elif choice == "0":
            # Esce dal ciclo while
            break

        else:
            # Stampa un messaggio di errore se l'utente ha inserito un'opzione non valida
            print("Opzione non valida. Riprova.")