import numpy as np

# Exo 1 : Analyse des ventes

print("Exercice 1 : ")

ventes = np.array([
    [120, 150, 180, 210],
    [100, 130, 170, 200],
    [140, 160, 190, 230]
])

magasins = np.array(["Magasin 1", "Magasin 2", "Magasin 3"])

# Lignes = magasins
# Colonnes = trimestres

# 1. Les ventes totales de chaque magasin.
ventes_totales_mag = np.sum(ventes, axis = 1)
print(" Les ventes totales de chaque magasin sont : ", ventes_totales_mag)

# 2. Les ventes totales de chaque trimestres.
ventes_totales_trimestres = np.sum(ventes, axis = 0)
print(" Les ventes totales de chaque trimestres sont : ", ventes_totales_trimestres)

#3. La moyenne des ventes par magasin.
vente_mean_magasin = np.mean(ventes, axis = 1)
print(" La moyenne des ventes par magasin est : ", vente_mean_magasin)

#La moyenne des ventes par trimestres.  (fait pour tester mes connaissances)
vente_mean_trimestres = np.mean(ventes, axis = 0)
print(" La moyenne des ventes par trimestres est : ", vente_mean_trimestres)

#4 Trouver le magasin ayant les ventes totales les plus élevées.
vente_totale_max_mag = np.argmax(ventes_totales_mag)
print(" Le magasin ayant les ventes totales les plus élevées est : ", vente_totale_max_mag)

#5 Trouver le trimestre le plus performant.
max_trim = np.argmax(ventes_totales_trimestres)
print(" Le trimestre le plus performant. est : ", max_trim)

# 6. Afficher uniquement les magasins ayant vendu plus de 650 unités.
mask_sup_650 = ventes_totales_mag > 650
print("Le masque est : ", mask_sup_650)
print("les magasins ayant vendu plus de 650 unités sont : ", magasins[mask_sup_650])

# 7. Ajouter un quatrième magasin
c = np.array([[110, 140, 175, 205]])
ventes_c = np.vstack((ventes, c))

#8. L'écart-type des ventes par trimestre.
vente_std_trimestre = np.std(ventes_c, axis = 0)
print(" L'écart-type des ventes par trimestre est : ", vente_std_trimestre)



# Exo 2 : Sélection conditionnelle

print("Exercice 2 : ")

notes = np.array([
    [70, 80, 90],
    [45, 55, 60],
    [95, 88, 92],
    [50, 65, 70]
])

# 1. Afficher toutes les notes supérieures à 80.
print(notes)
mask_notes_80 = notes > 80
print("Le masque est : ", mask_notes_80)
print("les notes supérieures à 80 sont : ", notes[mask_notes_80])

# 2. Combien de notes sont supérieures ou égales à 60 ?
mask_notes_60 = notes >= 60
print("Le masque est : ", mask_notes_60)
print("Le nombre de notes supérieures ou égales à 60 est :",
      np.sum(mask_notes_60))

# 3. Remplacer toutes les notes inférieures à 60 par 60.
mask_notes_inf_60 = notes < 60
notes[mask_notes_inf_60] = 60

# 4. Afficher les lignes dont la moyenne est supérieure à 80.
ligne_moy = np.mean(notes, axis = 1)
mask_ligne_moy_sup_80 = ligne_moy > 80
print("Les lignes dont la moyenne est supérieure à 80 sont :",
      notes[mask_ligne_moy_sup_80])


# 5. Créer un tableau booléen indiquant les notes supérieures à 80.
print(notes)
print("Le masque est : ", mask_notes_80)

# Exo 3 : Sélection conditionnelle

print("Exercice 3 : ")
tab = np.arange(1, 26).reshape(5,5)

print(tab)
# 1. Extraire la troisième ligne.
print(tab[2,:])

# 2. Extraire la quatrième colonne.
print(tab[:,3])

# 3. Extraire le carré central :
print(tab[1:4, 1:4])

# 4. Extraire les coins :
print(tab[[0, 0, 4, 4], [0, 4, 0, 4]])

# 5. Extraire la diagonale principale :
print(tab[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])



# Exo 4 : Produit matriciel

print("Exercice 4 : ")

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print("La somme de A et B est : ", A + B)
print("La soustraction de A et B est : ", A - B)
print("La Multiplication de A et B est : ", A * B)

print("Le produit matriciel de A et B est : ", np.dot(A,B))

# La différence entre A * B et np.dot(A,B) est que : 
# "A * B" est la multiplication entre les différentes colonnes des matrices et que 
# "np.dot(A,B)" c'est le produit matriciel de deux matrices

# Exo 5 : Statistiques multidimensionnelles

print("Exercice 5 : ")

data = np.array([
    [10, 20, 30, 40],
    [15, 25, 35, 45],
    [20, 30, 40, 50]
])


# Pour : axis=0
sum_0 = np.sum(data, axis = 0)
print("la somme sur 0 est : ", sum_0)

mean_0 = np.mean(data, axis = 0)
print("La moyenne sur 0 est : ", mean_0)

std_0 = np.std(data, axis = 0)
print("L'écart-type sur 0 est : ", std_0)

max_0 = np.max(data, axis = 0)
print("Le max sur 0 est : ", max_0)

min_0 = np.min(data, axis = 0)
print("Le min sur 0 est : ", min_0)


# Pour : axis=1
sum_1 = np.sum(data, axis = 1)
print("la somme sur 1 est : ", sum_1)

mean_1 = np.mean(data, axis = 1)
print("La moyenne sur 1 est : ", mean_1)

std_1 = np.std(data, axis = 1)
print("L'écart-type sur 1 est : ", std_1)

max_1 = np.max(data, axis = 1)
print("Le max sur 1 est : ", max_1)

min_1 = np.min(data, axis = 1)
print("Le min sur 1 est : ", min_1)



# Exo 6 : Challenge "Data Scientist Junior"

print("Exercice 6 : ")

employes = np.array([
    [25, 45000],
    [30, 52000],
    [35, 61000],
    [40, 72000],
    [50, 90000]
])

# Colonnes : âge | salaire

# 1. Calculer l'âge moyen.
age_moy = np.mean(employes[:, 0])
print("L'age moyen est : ", age_moy)

# 2. Calculer le salaire moyen.
salaire_moy = np.mean(employes[:, 1])
print("Le salaire moyen est : ", salaire_moy)

# 3. Trouver l'employé le plus âgé.
plus_age = np.argmax(employes[:, 0])
print("L'employé le plus agé est : ", plus_age)

# 4. Trouver le salaire maximal.
salaire_max = np.argmax(employes[:, 1])
print("Le salaire maximal est : ", salaire_max)

# 5. Afficher seulement les employés de plus de 35 ans.
plus_35 = employes[:, 0] > 35
print("le masque est : ", plus_35)
print("les employés de plus de 35 ans sont : ", employes[plus_35])

# 6. Afficher seulement les employés gagnant plus de 60 000.
sal_plus_60k = employes[:, 1] > 60000
print("le masque est : ", sal_plus_60k)
print("les employés gagnant plus de 60 000 sont : ", employes[sal_plus_60k])




# Exo 7 : Niveau avancé (3 dimensions)

print("Exercice 7 : ")

tab = np.array([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

# 1. Quelle est la valeur de :
print(tab.shape)

# 2. Quelle est la valeur de :
print(tab.ndim)

# 3. Afficher :
print(tab[0])

# 4. Afficher :
print(tab[1])

# 5. Afficher :
print(tab[0,1,1])
 
# 6. Afficher :
print(tab[:,:,0])

# 7. Afficher :
print(tab[:,:,1])


notes = np.array([70, 95, 60, 85])

indices = np.argsort(notes)

print(indices)



salaires = np.array([45000, 90000, 52000, 72000, 61000])
print("tri des salaires : ", np.argsort(salaires)[::-1][:3])
print("tri des salaires : ", salaires[np.argsort(salaires)[::-1][:3]])