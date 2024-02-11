from collections import Counter

def count_occurrences(clustering):
    # Utiliser Counter pour compter les occurrences de chaque valeur
    occurrences = Counter(clustering)
    
    # Convertir le résultat en une liste de tuples (valeur, nombre d'occurrences)
    occurrences_list = list(occurrences.items())
    
    # Trier la liste par valeurs pour obtenir une représentation ordonnée
    sorted_occurrences = sorted(occurrences_list, key=lambda x: x[0])
    
    # Diviser la liste en deux listes séparées (valeurs, nombre d'occurrences)
    values, counts = zip(*sorted_occurrences)
    
    return values, counts

# Exemple d'utilisation
clustering = [0, 3,5,5, 2, 3, 1, 3, 1, 2, 4,2, 3]
values, counts = count_occurrences(clustering)

print("Valeurs uniques :", values)
print("Nombre d'occurrences correspondant à chaque valeur :", counts)